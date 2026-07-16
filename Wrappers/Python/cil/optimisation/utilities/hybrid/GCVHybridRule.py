import numpy as np
import scipy.optimize
from scipy.ndimage import gaussian_filter1d
from typing import List
from dataclasses import dataclass, field

from .BaseHybridRule import BaseHybridRule
from .maths import projected_residual_norm_sq, KrylovState, find_optimal_alpha_via_grid


@dataclass
class GCVHistory:
    """Encapsulates history tracking specific to Generalized Cross-Validation."""

    omegas: List[float] = field(default_factory=list)
    Ghats: List[float] = field(default_factory=list)


class GCVHybridRule(BaseHybridRule):
    """
    Generalized Cross-Validation (GCV) stopping rule for iterative solvers.

    This rule determines the optimal regularization parameter by minimizing the
    GCV objective function at each iteration. It includes support for weighted
    and adaptive-weighted GCV formulations to handle highly correlated noise.
    """

    def __init__(
        self,
        data_size: int,
        domain_size: int,
        tol: float = 1e-2,
        gcv_weight: float = 1.0,
        adaptive_weight: bool = True,
    ):
        super().__init__(data_size, domain_size, tol)

        if adaptive_weight:
            gcv_type = "adaptive-weighted"
        elif gcv_weight == 1.0:
            gcv_type = "standard"
        else:
            gcv_type = "weighted"

        self.rule_type = f"{gcv_type} gcv"
        self.gcv_type = gcv_type

        # State variables
        self.initial_gcv_weight = gcv_weight
        self.omega = gcv_weight
        self.gcv_history = GCVHistory()

    def initialize(self, initial_alpha: float, initial_beta: float) -> None:
        """Overrides base to ensure GCV-specific history is also cleared on reset."""
        super().initialize(initial_alpha, initial_beta)
        self.gcv_history = GCVHistory()

        # Reset omega back to the original configured weight for a fresh run
        if self.gcv_type == "adaptive-weighted":
            self.omega = self.initial_gcv_weight

    def _calculate_optimal_regalpha(
        self, state: KrylovState, lower_bound: float, upper_bound: float
    ) -> float:
        """
        Executes a two-stage optimization (grid search followed by continuous minimization)
        to find the optimal regularization parameter.
        """
        if self.gcv_type == "adaptive-weighted":
            self.omega = self._adaptive_omega(state)

        bounds = (max(lower_bound, self.config.eps), upper_bound)

        def bound_objective(alpha: float) -> float:
            return self.evaluate_objective(alpha, state)

        search_result = find_optimal_alpha_via_grid(
            bound_objective,
            bounds=bounds,
            num_points=self.config.default_grid_points,
        )

        # Smooth the grid to find reliable gradients
        func_grid_smooth = gaussian_filter1d(search_result.func_grid, sigma=2)
        grad = np.gradient(func_grid_smooth, np.log(search_result.alpha_grid))

        grad_tolerance = self.config.grad_tolerance_multiplier * np.max(np.abs(grad))
        increasing_indices = np.where(grad > grad_tolerance)[0]

        # Seed the local optimization based on the gradient
        x0 = (
            search_result.alpha_grid[increasing_indices[0]]
            if len(increasing_indices) > 0
            else search_result.best_alpha
        )

        res = scipy.optimize.minimize(
            self.evaluate_objective,
            x0=x0,
            args=(state,),
            bounds=[bounds],
            tol=1e-10,
        )

        # Fallback to the best grid point if continuous minimization fails
        new_regalpha = (
            float(res.x[0]) if res.success and np.isfinite(res.x[0]) else float(x0)
        )

        # Record the Ghat value for convergence checking
        if state.iteration > 1:
            self.gcv_history.Ghats.append(self._Ghat_objective(new_regalpha, state))

        return new_regalpha

    def _adaptive_omega(self, state: KrylovState) -> float:
        """
        Computes an adaptive weight (omega) for the GCV denominator.

        This uses an estimator designed to robustly handle noise levels that change
        dynamically throughout the Krylov subspace iterations.
        """
        filt = 1.0 / (state.min_singular_value + state.singular_values_squared)
        u1_tail_sq = state.u1_tail * state.u1_tail

        num = (
            (state.iteration + 1)
            * state.min_singular_value**2
            * np.sum(u1_tail_sq * state.singular_values_squared * np.power(filt, 3))
        )
        denom1 = (
            np.sum(
                np.power(state.min_singular_value, 4) * u1_tail_sq * np.power(filt, 2)
            )
            + u1_tail_sq
        )
        denom2 = np.sum(state.singular_values_squared * np.power(filt, 2))
        denom3 = state.min_singular_value**2 * np.sum(
            u1_tail_sq * state.singular_values_squared * np.power(filt, 3)
        )
        denom4 = np.sum(state.singular_values_squared * filt)

        omega = float(num / (denom1 * denom2 + denom3 * denom4))
        self.gcv_history.omegas.append(omega)

        if (
            state.min_singular_value / state.max_singular_value
        ) < self.config.svd_ratio_tol:
            omega = float(np.mean(self.gcv_history.omegas))

        return omega

    def _weighted_trace(
        self, regalpha: float, constrained_dofs: int, state: KrylovState
    ) -> float:
        r"""
        Computes the weighted squared trace used in the denominator of the GCV function.

        The squared trace is calculated as:

        .. math::

            \text{Trace}(\alpha) = \left( c + \sum_{i=1}^k \frac{(1 - \omega)\sigma_i^2 + \alpha^2}{\sigma_i^2 + \alpha^2} \right)^2

        Where:
        * :math:`c` is the constrained degrees of freedom.
        * :math:`\omega` is the GCV weight parameter.
        * :math:`\sigma_i` are the singular values.
        """
        filt = ((1.0 - self.omega) * state.singular_values_squared + regalpha**2) / (
            state.singular_values_squared + regalpha**2
        )
        return float(np.square(constrained_dofs + np.sum(filt)))

    def evaluate_objective(self, regalpha: float, state: KrylovState) -> float:
        r"""
        Evaluates the standard projected GCV objective function for a given alpha.

        The objective minimizes the ratio of the projected residual norm to the trace:

        .. math::

            V(\alpha) = \frac{k \| r_k(\alpha) \|^2}{\text{Trace}(\alpha)}
        """
        return (
            state.iteration
            * projected_residual_norm_sq(regalpha, state, self.b_norm)
            / self._weighted_trace(regalpha, 1, state)
        )

    def _Ghat_objective(self, regalpha: float, state: KrylovState) -> float:
        r"""
        Evaluates the Generalized Cross-Validation parameter :math:`\hat{G}(\alpha)`.

        This formulation scales with the full domain size (:math:`n`) and is monitored
        across iterations to determine solver convergence:

        .. math::

            \hat{G}(\alpha) = \frac{n \| r_k(\alpha) \|^2}{\text{Trace}(\alpha)}

        Where the trace parameter :math:`c` is given by :math:`m - k` (data size minus iteration).
        """
        return (
            self.n
            * projected_residual_norm_sq(regalpha, state, self.b_norm)
            / self._weighted_trace(regalpha, self.m - state.iteration, state)
        )

    def _check_convergence(self) -> bool:
        """
        Overrides the base convergence check to integrate GCV-specific heuristics.
        """
        # 1. Standard Alpha saturation rule
        if super()._check_convergence():
            return True

        ghats = self.gcv_history.Ghats

        # 2. GCV specific rules based on the Ghat curve trajectory
        if len(ghats) > 3:
            denom = abs(ghats[0]) + self.config.eps

            # Saturation of the Ghat value
            if (abs(ghats[-1] - ghats[-2]) / denom) < self.tol:
                return True

            # Upward curvature: Stop if the GCV curve starts turning upwards
            elif ghats[-1] > ghats[-2] > ghats[-3]:
                return True

        return False
