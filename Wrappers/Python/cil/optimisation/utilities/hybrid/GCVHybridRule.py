import numpy as np
import scipy.optimize
from scipy.ndimage import gaussian_filter1d
from typing import List
from dataclasses import dataclass, field

from .BaseHybridRule import BaseHybridRule, RuleConfig
from typing import Optional
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
        tol: float = 1e-4,
        gcv_weight: float = 1.0,
        adaptive_weight: bool = True,
        config: Optional[RuleConfig] = None,
    ):
        super().__init__(tol, config)

        if adaptive_weight:
            gcv_type = "adaptive-weighted"
        elif gcv_weight == 1.0:
            gcv_type = "standard"
        else:
            gcv_type = "weighted"

        self.rule_type = f"{gcv_type} gcv"
        self.gcv_type = gcv_type

        self.initial_gcv_weight = gcv_weight
        self.omega = gcv_weight
        self.gcv_history = GCVHistory()

    def reset_state(self, initial_alpha: float, initial_beta: float) -> None:
        super().reset_state(initial_alpha, initial_beta)
        self.gcv_history = GCVHistory()

        if self.gcv_type == "adaptive-weighted":
            self.omega = self.initial_gcv_weight

    def _calculate_optimal_regalpha(
        self, state: KrylovState, lower_bound: float, upper_bound: float
    ) -> float:
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

        func_grid_smooth = gaussian_filter1d(search_result.func_grid, sigma=2)
        safe_alphas = np.clip(search_result.alpha_grid, self.config.eps, None)
        grad = np.gradient(func_grid_smooth, np.log(safe_alphas))

        grad_tolerance = self.config.grad_tolerance_multiplier * np.max(np.abs(grad))
        increasing_indices = np.where(grad > grad_tolerance)[0]

        x0 = (
            search_result.alpha_grid[increasing_indices[0]]
            if len(increasing_indices) > 0
            else search_result.best_alpha
        )
        x0 = float(np.clip(x0, bounds[0], bounds[1]))

        res = scipy.optimize.minimize(
            self.evaluate_objective,
            x0=x0,
            args=(state,),
            bounds=[bounds],
            tol=1e-10,
        )

        new_regalpha = (
            float(res.x[0]) if res.success and np.isfinite(res.x[0]) else float(x0)
        )

        if state.iteration > 1:
            self.gcv_history.Ghats.append(self._Ghat_objective(new_regalpha, state))

        return new_regalpha

    def _adaptive_omega(self, state: KrylovState) -> float:
        """
        Computes an adaptive weight (omega) for the GCV denominator.
        """
        # BUGFIX 1: Square the minimum singular value to match dimensional variance
        min_sig_sq = state.min_singular_value**2

        filt = 1.0 / (min_sig_sq + state.singular_values_squared)

        # BUGFIX 2: Use the spectral components (state.u1_squared) for the sums,
        # saving the tail only for the irreducible residual term.
        u1_sq = state.u1_squared
        u1_tail_sq = state.u1_tail**2

        num = (
            (state.iteration + 1)
            * min_sig_sq
            * np.sum(u1_sq * state.singular_values_squared * np.power(filt, 3))
        )

        denom1 = np.sum((min_sig_sq**2) * u1_sq * np.power(filt, 2)) + u1_tail_sq

        denom2 = np.sum(state.singular_values_squared * np.power(filt, 2))

        denom3 = min_sig_sq * np.sum(
            u1_sq * state.singular_values_squared * np.power(filt, 3)
        )

        denom4 = np.sum(state.singular_values_squared * filt)

        # Protect against division by zero
        denom = denom1 * denom2 + denom3 * denom4
        if denom < self.config.eps:
            omega = (
                float(np.mean(self.gcv_history.omegas))
                if self.gcv_history.omegas
                else 1.0
            )
        else:
            omega = float(num / denom)

        self.gcv_history.omegas.append(omega)

        if (
            state.min_singular_value / state.max_singular_value
        ) < self.config.svd_ratio_tol:
            omega = float(np.mean(self.gcv_history.omegas))

        return omega

    def _weighted_trace(
        self, regalpha: float, constrained_dofs: int, state: KrylovState
    ) -> float:
        filt = ((1.0 - self.omega) * state.singular_values_squared + regalpha**2) / (
            state.singular_values_squared + regalpha**2
        )
        return float(np.square(constrained_dofs + np.sum(filt)))

    def evaluate_objective(self, regalpha: float, state: KrylovState) -> float:
        # Adjusted to state.iteration + 1 to correctly match projected DoFs (m_proj)
        return (
            (state.iteration + 1)
            * projected_residual_norm_sq(regalpha, state, self.b_norm)
            / self._weighted_trace(regalpha, 1, state)
        )

    def _Ghat_objective(self, regalpha: float, state: KrylovState) -> float:
        return (
            self.n
            * projected_residual_norm_sq(regalpha, state, self.b_norm)
            / self._weighted_trace(regalpha, self.m - state.iteration, state)
        )

    def _check_convergence(self) -> bool:
        if super()._check_convergence():
            return True

        ghats = self.gcv_history.Ghats

        if len(ghats) > 3:
            denom = abs(ghats[0]) + self.config.eps

            if (abs(ghats[-1] - ghats[-2]) / denom) < self.tol:
                return True

            elif ghats[-1] > ghats[-2] > ghats[-3]:
                return True

        return False
