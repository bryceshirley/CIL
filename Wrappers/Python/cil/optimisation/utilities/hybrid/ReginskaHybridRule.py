import numpy as np
import scipy.optimize

from .BaseHybridRule import BaseHybridRule
from .maths import (
    projected_residual_norm_sq, 
    projected_solution_norm_sq, 
    projected_norm_first_derivatives,
    KrylovState, 
    find_optimal_alpha_via_grid
)


class ReginskaHybridRule(BaseHybridRule):
    """
    Reginska's stopping rule for iterative solvers.
    
    This rule selects the regularisation parameter by minimizing a heuristic 
    function related to the L-curve. It balances the residual norm and the 
    solution norm in a log-log scale.
    """

    def __init__(
        self, 
        data_size: int, 
        domain_size: int, 
        tol: float = 1e-2, 
        mu: float = 0.5
    ):
        super().__init__(data_size, domain_size, tol)
        self.rule_type = "reginska"
        self.mu = mu

    def _calculate_optimal_regalpha(
        self, state: KrylovState, lower_bound: float, upper_bound: float
    ) -> float:
        """
        Executes a two-stage optimization using analytical derivatives 
        to find the optimal regularization parameter.
        """
        bounds = (max(lower_bound, self.config.eps), upper_bound)
        
        # 1. Local closures to inject state cleanly (No lambdas!)
        def bound_objective(alpha: float) -> float:
            return self.evaluate_objective(alpha, state)

        def bound_derivative(alpha: float) -> float:
            return self.evaluate_derivative(alpha, state)

        # 2. Grid search with derivative sign-change detection
        search_result = find_optimal_alpha_via_grid(
            bound_objective, 
            dfunc=bound_derivative, 
            bounds=bounds,
            num_points=self.config.default_grid_points
        )

        # 3. Continuous bounded minimization (utilizing the analytical Jacobian)
        res = scipy.optimize.minimize(
            self.evaluate_objective,
            x0=search_result.best_alpha,
            args=(state,),
            jac=self.evaluate_derivative, # Explicitly pass the analytical derivative
            bounds=[bounds],
            tol=1e-10
        )
        
        # 4. Return result or fallback to grid search minimum
        if res.success and np.isfinite(res.x[0]):
            return float(res.x[0])
            
        return float(search_result.best_alpha)

    def evaluate_objective(self, regalpha: float, state: KrylovState) -> float:
        r"""
        Evaluates Reginska's objective function for a given alpha.

        The function is defined as:

        .. math::

            V(\alpha) = \frac{1}{2} \left( \ln \| r_k(\alpha) \|^2 + \mu \ln \| x_k(\alpha) \|^2 \right)

        Where :math:`\mu` is typically 0.5 (corresponding to the L-curve corner).
        """
        r2 = projected_residual_norm_sq(regalpha, state, self.b_norm)
        x2 = projected_solution_norm_sq(regalpha, state, self.b_norm)
        
        # Prevent log(0) domain errors
        if r2 <= self.config.eps or x2 <= self.config.eps:
            return 1e30 
            
        return float(0.5 * (np.log(r2) + self.mu * np.log(x2)))

    def evaluate_derivative(self, regalpha: float, state: KrylovState) -> float:
        r"""
        Evaluates the analytical first derivative of Reginska's objective function.

        By the chain rule, the derivative with respect to :math:`\alpha` is:

        .. math::

            V'(\alpha) = \frac{1}{2} \left( \frac{(R^2)'}{R^2} + \mu \frac{(X^2)'}{X^2} \right)
        """
        R2 = projected_residual_norm_sq(regalpha, state, self.b_norm)
        X2 = projected_solution_norm_sq(regalpha, state, self.b_norm)
        R2_p, X2_p = projected_norm_first_derivatives(regalpha, state, self.b_norm)

        # Protect denominators
        denom_r = max(R2, self.config.eps)
        denom_x = max(X2, self.config.eps)

        return float(0.5 * (R2_p / denom_r + self.mu * X2_p / denom_x))