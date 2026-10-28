#  Copyright 2026 United Kingdom Research and Innovation
#  Copyright 2026 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at:
# https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import logging

import numpy as np
import scipy.optimize

from .HybridRule import HybridRule, projected_residual_norm_sq

log = logging.getLogger(__name__)


class DiscrepHybridRule(HybridRule):
    r"""
    Morozov's discrepancy principle for the regularisation parameter of a
    hybrid LSQR run: :math:`\alpha` is chosen so that
    :math:`\|A x_\alpha - b\|_2^2 = \delta^2`.

    Parameters
    ----------
    noise_level_estimate : float
        The noise norm :math:`\delta`, positive.
    tol : float, default 1e-3
        As in :class:`HybridRule`.
    mode : {'stop', 'match', 'select_only'}, default 'stop'
        As in :class:`HybridRule`.
    """

    def __init__(self, noise_level_estimate, tol=1e-3, mode='stop'):
        # With delta = 0 the residual can never reach the target and the
        # rule would never select.
        if noise_level_estimate <= 0:
            raise ValueError(
                "noise_level_estimate must be positive, got "
                f"{noise_level_estimate}.")
        super().__init__(tol, mode=mode)
        self.noise_level_estimate = noise_level_estimate

    def _calculate_optimal_regalpha(self, state, upper_bound):
        f_lo = self.evaluate_objective(self.EPS, state)
        f_hi = self.evaluate_objective(upper_bound, state)

        # Early iterations cannot reach the target residual yet.
        if f_lo > 0:
            return np.nan

        if f_hi < 0:
            log.debug("The noise level exceeds the projected data norm.")
            return np.nan

        result = scipy.optimize.root_scalar(
            self.evaluate_objective,
            args=(state,),
            bracket=[self.EPS, upper_bound],
            method="brentq",
        )

        return float(result.root)

    def evaluate_objective(self, regalpha, state):
        r"""The discrepancy :math:`\|r(\alpha)\|^2 - \delta^2` of the projected problem."""
        return (
            projected_residual_norm_sq(regalpha, state, self.b_norm)
            - self.noise_level_estimate**2
        )
