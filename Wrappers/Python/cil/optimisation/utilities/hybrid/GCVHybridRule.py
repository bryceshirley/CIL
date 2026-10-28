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

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .HybridRule import (HybridRule, find_optimal_alpha_via_grid,
                         projected_residual_norm_sq, refine_bracket)


class GCVHistory:
    """History specific to Generalized Cross-Validation."""

    def __init__(self):
        self.omegas = []
        self.Ghats = []


class GCVHybridRule(HybridRule):
    r"""
    Generalised cross-validation (GCV) rule for the regularisation parameter
    of a hybrid LSQR run: :math:`\alpha` minimises the GCV function of the
    projected problem.

    The weighted form replaces the trace term with
    :math:`\mathrm{trace}(I - \omega A A_\alpha^\dagger)`; a weight below 1
    shifts the selection towards smaller regularisation than plain GCV.

    In the adaptive form the weight is re-estimated each iteration, clamped
    to ``[0, 1]`` and averaged over the iterations, as in HyBR; when the
    estimates saturate at 1 the rule coincides with standard GCV.

    Parameters
    ----------
    tol : float, default 1e-3
        As in :class:`HybridRule`.
    gcv_weight : float, default 1.0
        The weight :math:`\omega`; ignored when `adaptive_weight` is True.
    adaptive_weight : bool, default True
        Re-estimate :math:`\omega` each iteration, as in HyBR.
    mode : {'stop', 'match', 'select_only'}, default 'stop'
        As in :class:`HybridRule`.

    Reference
    ---------
    - Chung, J., Nagy, J.G. and O'Leary, D.P., 2008. A weighted-GCV method
      for Lanczos-hybrid regularization. ETNA 28, 149-167.
    """

    # Relative gradient threshold of the basin pick in the flat-valley
    # heuristic below.
    GRAD_TOLERANCE_MULTIPLIER = 0.01

    def __init__(self, tol=1e-3, gcv_weight=1.0, adaptive_weight=True,
                 mode='stop'):
        super().__init__(tol, mode=mode)
        self.adaptive_weight = adaptive_weight
        self.omega = gcv_weight
        self.gcv_history = GCVHistory()

    def seed(self, algorithm=None, initial_alpha=None, initial_beta=None,
             data_size=None, domain_size=None):
        super().seed(algorithm=algorithm, initial_alpha=initial_alpha, initial_beta=initial_beta,
             data_size=data_size, domain_size=domain_size)
        self.gcv_history = GCVHistory()

    def _calculate_optimal_regalpha(self, state, upper_bound):
        if self.adaptive_weight:
            self.omega = self._adaptive_omega(state)

        bounds = (self.EPS, upper_bound)

        def bound_objective(alpha):
            return self.evaluate_objective(alpha, state)

        search_result = find_optimal_alpha_via_grid(
            bound_objective,
            bounds=bounds,
            num_points=self.GRID_POINTS,
            refine=False,
        )
        if not np.isfinite(search_result.best_alpha):
            return np.nan

        # The GCV valley can stay flat towards alpha -> 0: pick the basin
        # where the smoothed grid first turns upward.
        func_grid_smooth = gaussian_filter1d(search_result.func_grid, sigma=2)
        grad = np.gradient(func_grid_smooth, np.log(search_result.alpha_grid))

        grad_tolerance = self.GRAD_TOLERANCE_MULTIPLIER * np.max(np.abs(grad))
        increasing_indices = np.where(grad > grad_tolerance)[0]

        index = (
            int(increasing_indices[0])
            if len(increasing_indices) > 0
            else search_result.best_index
        )

        new_regalpha = refine_bracket(bound_objective,
                                      search_result.alpha_grid, index)

        # A minimum parked at the grid floor, or far below the resolved
        # spectrum, is vanishing regularisation rather than a selection:
        # report none, so a driver keeps its previous selection instead.
        if (new_regalpha <= 1.05 * search_result.alpha_grid[0]
                or new_regalpha < 0.01 * state.min_singular_value):
            return np.nan

        if state.iteration > 1 and self.m is not None and self.n is not None:
            # Record Ghat only for resolved selections; a selection in the
            # unresolved top of the spectrum tracks the wrong basin.
            if self._selection_is_resolved(state, pick=new_regalpha):
                self.gcv_history.Ghats.append(
                    self._Ghat_objective(new_regalpha, state))

        return new_regalpha

    def _adaptive_omega(self, state):
        r"""HyBR estimate of the GCV weight: the :math:`\omega` making
        :math:`\sigma_{min}(B_k)` a stationary point of :math:`G_\omega`,
        clamped to ``[0, 1]`` and averaged over the iterations."""
        min_sig_sq = state.min_singular_value**2

        filt = 1.0 / (min_sig_sq + state.singular_values_squared)

        # The tail only enters the irreducible residual term.
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

        denom = denom1 * denom2 + denom3 * denom4
        if denom < self.EPS:
            omega = (
                float(np.mean(self.gcv_history.omegas))
                if self.gcv_history.omegas
                else 1.0
            )
        else:
            omega = float(num / denom)

        self.gcv_history.omegas.append(min(1.0, max(0.0, omega)))

        return float(np.mean(self.gcv_history.omegas))

    def _trace_squared(self, regalpha, constrained_dofs, state):
        """The squared trace of I - A(regalpha), the GCV denominator."""
        filt = ((1.0 - self.omega) * state.singular_values_squared + regalpha**2) / (
            state.singular_values_squared + regalpha**2
        )
        return float(np.square(constrained_dofs + np.sum(filt)))

    def evaluate_objective(self, regalpha, state):
        # The projected problem has iteration + 1 rows.
        return (
            (state.iteration + 1)
            * projected_residual_norm_sq(regalpha, state, self.b_norm)
            / self._trace_squared(regalpha, 1, state)
        )

    def _Ghat_objective(self, regalpha, state):
        return (
            self.n
            * projected_residual_norm_sq(regalpha, state, self.b_norm)
            / self._trace_squared(regalpha, self.m - state.iteration, state)
        )

    def _pick_has_settled(self):
        """True when the finite selections over the last
        WINDOW_STABLE_LOOKBACK iterations are all within `tol` of the
        current one."""
        lookback = self.WINDOW_STABLE_LOOKBACK
        picks = [p for p in self.history.regalphas[-lookback:]
                 if np.isfinite(p)]
        if len(picks) < 2 or not np.isfinite(self.selected_alpha):
            return False
        denom = abs(self.selected_alpha) + self.EPS
        return all(abs(p - self.selected_alpha) / denom <= self.tol
                   for p in picks)

    def _check_convergence(self):
        # A selection at or below sigma_min(B_k) while the window still
        # expands tracks the unresolved spectrum, not an optimum.
        if (self._window_lows
                and np.isfinite(self.selected_alpha)
                and self.selected_alpha <= self._window_lows[-1]
                and not self._window_has_settled()):
            return False

        if super()._check_convergence():
            return True

        # The Ghat shortcuts share the preconditions of the base check:
        # on an unresolved subspace a rising Ghat is an artifact.
        if not self._pick_is_assessable():
            return False

        ghats = self.gcv_history.Ghats

        # Ghat trends only count once the window has stopped growing and
        # the selection has been still over the same lookback.
        if (len(ghats) > 3 and self._window_has_settled()
                and self._pick_has_settled()):
            denom = abs(ghats[0]) + self.EPS

            if (abs(ghats[-1] - ghats[-2]) / denom) < self.tol:
                return True

            elif ghats[-1] > ghats[-2] > ghats[-3]:
                return True

        return False
