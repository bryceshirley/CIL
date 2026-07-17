import numpy as np
import scipy.optimize

from .BaseHybridRule import BaseHybridRule, RuleConfig
from typing import Optional
from .maths import (
    projected_residual_norm_sq,
    projected_solution_norm_sq,
    projected_norm_first_derivatives,
    projected_norm_second_derivatives,
    projected_norm_third_derivatives,
    KrylovState,
    find_optimal_alpha_via_grid,
)


class LCurveHybridRule(BaseHybridRule):
    """
    L-Curve stopping rule for iterative solvers.

    This rule selects the regularization parameter by finding the point of
    maximum curvature on the L-curve, representing the optimal trade-off
    between the residual norm and the solution norm.
    """

    def __init__(self, tol: float = 1e-2, config: Optional[RuleConfig] = None):
        super().__init__(tol, config)
        self.rule_type = "l-curve"

    def _calculate_optimal_regalpha(
        self, state: KrylovState, lower_bound: float, upper_bound: float
    ) -> float:
        bounds = (max(lower_bound, self.config.eps), upper_bound)

        def bound_objective(alpha: float) -> float:
            return self.evaluate_objective(alpha, state)

        def bound_derivative(alpha: float) -> float:
            return self.evaluate_derivative(alpha, state)

        search_result = find_optimal_alpha_via_grid(
            bound_objective,
            dfunc=bound_derivative,
            bounds=bounds,
            num_points=self.config.default_grid_points,
        )

        res = scipy.optimize.minimize(
            self.evaluate_objective,
            x0=search_result.best_alpha,
            args=(state,),
            jac=self.evaluate_derivative,
            bounds=[bounds],
            tol=1e-10,
        )

        if res.success and np.isfinite(res.x[0]):
            return float(res.x[0])
        return float(search_result.best_alpha)

    def evaluate_objective(self, regalpha: float, state: KrylovState) -> float:
        r"""
        Computes the negative curvature of the L-curve in log-log scale.
        """
        R2 = projected_residual_norm_sq(regalpha, state, self.b_norm)
        X2 = projected_solution_norm_sq(regalpha, state, self.b_norm)

        if R2 <= self.config.eps or X2 <= self.config.eps:
            return 0.0

        R2_p, X2_p = projected_norm_first_derivatives(regalpha, state, self.b_norm)
        R2_pp, X2_pp = projected_norm_second_derivatives(regalpha, state, self.b_norm)

        # Log derivatives
        logR_p = 0.5 * R2_p / R2
        logX_p = 0.5 * X2_p / X2
        logR_pp = (R2 * R2_pp - R2_p**2) / (2 * R2**2)
        logX_pp = (X2 * X2_pp - X2_p**2) / (2 * X2**2)

        # Curvature formula
        num = logR_p * logX_pp - logX_p * logR_pp
        denom = (logR_p**2 + logX_p**2) ** 1.5 + 1e-300

        return float(-num / denom)

    def evaluate_derivative(self, regalpha: float, state: KrylovState) -> float:
        r"""
        Evaluates the analytical derivative of the L-curve curvature.
        """
        R2 = projected_residual_norm_sq(regalpha, state, self.b_norm)
        X2 = projected_solution_norm_sq(regalpha, state, self.b_norm)

        R2_p, X2_p = projected_norm_first_derivatives(regalpha, state, self.b_norm)
        R2_pp, X2_pp = projected_norm_second_derivatives(regalpha, state, self.b_norm)
        R2_ppp, X2_ppp = projected_norm_third_derivatives(regalpha, state, self.b_norm)

        # rho and eta derivatives
        rho_p = 0.5 * R2_p / R2
        rho_pp = (R2 * R2_pp - R2_p**2) / (2 * R2**2)
        rho_ppp = (R2 * R2_ppp - 3 * R2_p * R2_pp + 2 * R2_p**3 / R2) / (2 * R2**2)

        eta_p = 0.5 * X2_p / X2
        eta_pp = (X2 * X2_pp - X2_p**2) / (2 * X2**2)
        eta_ppp = (X2 * X2_ppp - 3 * X2_p * X2_pp + 2 * X2_p**3 / X2) / (2 * X2**2)

        # Derivative of curvature formula
        num = (rho_p * eta_ppp - eta_p * rho_ppp) * (rho_p**2 + eta_p**2)
        denom = (rho_p**2 + eta_p**2) ** (2.5)
        correction = (
            3 * (rho_p * eta_pp - eta_p * rho_pp) * (rho_p * rho_pp + eta_p * eta_pp)
        )

        return float((num - correction) / denom)
