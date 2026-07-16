import scipy.optimize
from .BaseHybridRule import BaseHybridRule
from .maths import projected_residual_norm_sq, KrylovState
import numpy as np


class DiscrepHybridRule(BaseHybridRule):
    def __init__(
        self,
        data_size: int,
        domain_size: int,
        tol: float = 1e-2,
        noise_level_estimate=0.0,
    ):
        super().__init__(data_size, domain_size, tol)
        self.rule_type = "discrep"
        self.noise_level_estimate = noise_level_estimate

    def _calculate_optimal_regalpha(
        self, state: KrylovState, lower_bound: float, upper_bound: float
    ) -> float:
        f_lo = self.evaluate_objective(lower_bound, state)
        f_hi = self.evaluate_objective(upper_bound, state)

        if f_lo > 0 or f_hi < 0:
            return np.nan

        result = scipy.optimize.root_scalar(
            self.evaluate_objective,
            args=(state,),
            bracket=[lower_bound, upper_bound],
            method="brentq",
        )
        return result.root if result.converged else np.nan

    def evaluate_objective(self, regalpha: float, state: KrylovState) -> float:
        return (
            projected_residual_norm_sq(regalpha, state, self.b_norm)
            - self.noise_level_estimate**2
        )
