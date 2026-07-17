import logging
import scipy.optimize
import numpy as np

from .BaseHybridRule import BaseHybridRule, RuleConfig
from typing import Optional
from .maths import projected_residual_norm_sq, KrylovState

log = logging.getLogger(__name__)


class DiscrepHybridRule(BaseHybridRule):
    """
    Morozov's Discrepancy Principle for choosing the regularisation parameter.

    Selects alpha such that the residual norm matches the known noise level:
    || A x_alpha - b ||^2 = delta^2
    """

    def __init__(
        self,
        tol: float = 1e-4,
        noise_level_estimate: float = 0.0,
        config: Optional[RuleConfig] = None,
    ):
        super().__init__(tol, config)
        self.rule_type = "discrep"
        self.noise_level_estimate = noise_level_estimate

    def _calculate_optimal_regalpha(
        self, state: KrylovState, lower_bound: float, upper_bound: float
    ) -> float:
        f_lo = self.evaluate_objective(lower_bound, state)
        f_hi = self.evaluate_objective(upper_bound, state)

        # 1. Evaluate Bracket Physics
        if f_lo > 0:
            log.warning(
                "Discrepancy rule failed: Noise estimate is too small. "
                f"Minimum achievable residual variance is {f_lo + self.noise_level_estimate**2:.2e}"
            )
            return np.nan

        if f_hi < 0:
            log.warning(
                "Discrepancy rule failed: Noise estimate exceeds total signal energy. "
                f"Data variance is {f_hi + self.noise_level_estimate**2:.2e}"
            )
            return np.nan

        # 2. Root finding via Brent's Method
        result = scipy.optimize.root_scalar(
            self.evaluate_objective,
            args=(state,),
            bracket=[lower_bound, upper_bound],
            method="brentq",
        )

        return float(result.root) if result.converged else np.nan

    def evaluate_objective(self, regalpha: float, state: KrylovState) -> float:
        """
        f(alpha) = || r(alpha) ||^2 - delta^2
        """
        return (
            projected_residual_norm_sq(regalpha, state, self.b_norm)
            - self.noise_level_estimate**2
        )
