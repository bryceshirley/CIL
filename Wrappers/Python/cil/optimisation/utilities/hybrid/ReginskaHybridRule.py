import numpy as np
import scipy.optimize

from .BaseHybridRule import BaseHybridRule, RuleConfig
from typing import Optional
from .maths import (
    projected_residual_norm_sq,
    projected_solution_norm_sq,
    projected_norm_first_derivatives,
    KrylovState,
    find_optimal_alpha_via_grid,
)


class ReginskaHybridRule(BaseHybridRule):
    """
    Reginska's stopping rule for iterative solvers.

    This rule selects the regularisation parameter by minimizing a heuristic
    function related to the L-curve. It balances the residual norm and the
    solution norm in a log-log scale.

    Parameters
    ----------
    tol : float, optional
        Tolerance for the optimization process. Default is 1e-4.
    mu : float, optional
        Weighting factor for the solution norm in the objective function which balances
        the trade-off between the residual and solution norms.
        Default is 0.5, which corresponds to the L-curve corner.
        If mu is larger, the solution norm is weighted more heavily, and if smaller,
        the residual norm is weighted more heavily.
    config : Optional[RuleConfig], optional
        Configuration object for the hybrid rule. If None, default settings are used.
    """

    def __init__(
        self, tol: float = 1e-4, mu: float = 0.5, config: Optional[RuleConfig] = None
    ):
        super().__init__(tol, config)
        self.rule_type = "reginska"
        self.mu = mu
    
    def _calculate_optimal_regalpha(
        self, state: KrylovState, lower_bound: float, upper_bound: float
    ) -> float:
        bounds = (max(lower_bound, self.config.eps), upper_bound)

        def bound_objective(alpha: float) -> float:
            return -self.evaluate_objective(alpha, state)

        def bound_derivative(alpha: float) -> float:
            return -self.evaluate_derivative(alpha, state)

        # Grid search with derivative sign-change detection
        search_result = find_optimal_alpha_via_grid(
            bound_objective,
            dfunc=bound_derivative,
            bounds=bounds,
            num_points=self.config.default_grid_points,
        )

        res = scipy.optimize.minimize(
            bound_objective,
            x0=search_result.best_alpha,
            args=(),
            jac=bound_derivative,  
            bounds=[bounds],
            tol=1e-10,
        )

        if res.success and np.isfinite(res.x[0]):
            return float(res.x[0])

        return float(search_result.best_alpha)

    def evaluate_objective(self, regalpha: float, state: KrylovState) -> float:
        r"""Evaluates Reginska's objective function for a given alpha."""
        
        # Smoothly clamp to prevent log(0) without breaking the optimizer
        r2 = max(projected_residual_norm_sq(regalpha, state, self.b_norm), 1e-30)
        x2 = max(projected_solution_norm_sq(regalpha, state, self.b_norm), 1e-30)

        return float(0.5 * (np.log(r2) + self.mu * np.log(x2)))

    def evaluate_derivative(self, regalpha: float, state: KrylovState) -> float:
        r"""Evaluates the analytical first derivative of Reginska's objective function."""
        
        # Smoothly clamp denominators
        R2 = max(projected_residual_norm_sq(regalpha, state, self.b_norm), 1e-30)
        X2 = max(projected_solution_norm_sq(regalpha, state, self.b_norm), 1e-30)
        
        R2_p, X2_p = projected_norm_first_derivatives(regalpha, state, self.b_norm)

        return float(0.5 * (R2_p / R2 + self.mu * X2_p / X2))