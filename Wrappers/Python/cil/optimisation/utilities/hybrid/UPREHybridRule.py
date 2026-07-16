import numpy as np
import scipy.optimize

from .BaseHybridRule import BaseHybridRule
from .maths import projected_residual_norm_sq, KrylovState, find_optimal_alpha_via_grid


class UPREHybridRule(BaseHybridRule):
    """
    Unbiased Predictive Risk Estimator (UPRE) stopping rule.
    
    This rule selects the regularisation parameter by minimizing an unbiased 
    estimator of the predictive risk. It requires a good estimate of the 
    data noise variance.
    """
    
    def __init__(
        self, 
        data_size: int, 
        domain_size: int, 
        tol: float = 1e-2, 
        noise_variance: float = 0.0
    ):
        super().__init__(data_size, domain_size, tol)
        self.rule_type = "upre"
        self.sigma2 = noise_variance

    def _calculate_optimal_regalpha(
        self, state: KrylovState, lower_bound: float, upper_bound: float
    ) -> float:
        """
        Executes a two-stage optimization (grid search followed by continuous minimization)
        to find the optimal regularization parameter.
        """
        bounds = (max(lower_bound, self.config.eps), upper_bound)
        

        def bound_objective(alpha: float) -> float:
            return self.evaluate_objective(alpha, state)

        search_result = find_optimal_alpha_via_grid(
            bound_objective, 
            bounds=bounds,
            num_points=self.config.default_grid_points
        )

        res = scipy.optimize.minimize(
            self.evaluate_objective,
            x0=search_result.best_alpha,
            args=(state,),
            bounds=[bounds],
            tol=1e-10
        )
        
        # Fallback to the best grid point if continuous minimization fails
        if res.success and np.isfinite(res.x[0]):
            return float(res.x[0])
            
        return float(search_result.best_alpha)

    def evaluate_objective(self, regalpha: float, state: KrylovState) -> float:
        r"""
        Evaluates the UPRE objective function for a given alpha.

        The UPRE function is defined as:

        .. math::

            U(\alpha) = \frac{1}{m} \| r_k(\alpha) \|^2 + \frac{2\sigma^2}{m} \text{Trace}(A(\alpha)) - \sigma^2

        Where the trace of the influence matrix is approximated using the filter factors:

        .. math::

            \text{Trace}(A(\alpha)) = \sum_{i=1}^k \frac{\sigma_i^2}{\sigma_i^2 + \alpha^2}
        """
        # 1. Projected residual norm squared
        r2 = projected_residual_norm_sq(regalpha, state, self.b_norm)
        
        # 2. Trace of the influence matrix
        s2 = state.singular_values_squared
        trace_A = np.sum(s2 / (s2 + regalpha**2))
        
        # 3. Assemble UPRE
        return float((1.0 / self.m) * r2 + (2.0 * self.sigma2 / self.m) * trace_A - self.sigma2)