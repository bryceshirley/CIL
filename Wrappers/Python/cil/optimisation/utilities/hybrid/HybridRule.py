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
import scipy.linalg
import scipy.optimize
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

class IterationHistory:
    """One pass's history: the GKB scalars and the per-iteration selections."""

    def __init__(self):
        self.gkb_alphas = []
        self.gkb_betas = []
        self.regalphas = []

def build_bidiagonal(gkb_alphas, gkb_betas):
    r"""
    The lower bidiagonal :math:`B_k` of a GKB scalar history. After
    seeding and :math:`j` update pairs, the projected matrix matching the
    solver's iterate :math:`x_j` is the :math:`(j+1) \times j`

    .. math::
        B_j = \mathrm{bidiag}(\alpha_1..\alpha_j;\ \beta_2..\beta_{j+1});

    the newest :math:`\alpha_{j+1}` belongs to the next column and is not
    included.
    """
    k = len(gkb_alphas) - 1
    Bk = np.zeros((k + 1, k))
    idx = np.arange(k)
    Bk[idx, idx] = gkb_alphas[:k]
    if k > 0:
        Bk[idx + 1, idx] = gkb_betas[1:k + 1]
    return Bk


class KrylovState:
    """The SVD of the projected matrix B_k, with the quantities the rules read."""

    def __init__(self, bk):
        u, s, vt = scipy.linalg.svd(bk, full_matrices=False)
        self.singular_values = s
        self.singular_values_squared = np.square(s)
        self.u1 = u[0, :]
        self.u1_squared = np.square(self.u1)
        # The e1 component outside Range(B_k): the first row of the full
        # U has unit norm, so the missing entry is determined up to sign,
        # and every use squares it.
        self.u1_tail = float(np.sqrt(max(0.0, 1.0 - float(np.sum(self.u1_squared)))))
        self.vt = vt
        self.max_singular_value = s[0]
        self.min_singular_value = s[-1]
        self.iteration = len(s)


class GridSearchResult:
    """The result of a 1D grid search optimisation."""

    def __init__(self, best_alpha, func_grid, alpha_grid, best_index=-1):
        self.best_alpha = best_alpha
        self.func_grid = func_grid
        self.alpha_grid = alpha_grid
        self.best_index = best_index


def refine_bracket(func, alpha_grid, index):
    """
    Bounded Brent solve in log10(alpha) between the grid neighbours of
    ``index``. The parameter spans many decades, so the solve runs in log
    space; the bracket keeps a multimodal objective in the basin the grid
    chose. Falls back to the grid point if the refined value is no better.
    """
    lo = alpha_grid[max(index - 1, 0)]
    hi = alpha_grid[min(index + 1, len(alpha_grid) - 1)]
    result = scipy.optimize.minimize_scalar(
        lambda t: func(10.0 ** t),
        bounds=(np.log10(lo), np.log10(hi)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    refined = 10.0 ** float(result.x)

    at_grid = func(alpha_grid[index])
    at_refined = func(refined)
    if np.isfinite(at_refined) and (not np.isfinite(at_grid)
                                    or at_refined <= at_grid):
        return refined
    return float(alpha_grid[index])


def find_optimal_alpha_via_grid(func, bounds, num_points=200, refine=True):
    """
    Search over a geometric grid, then refine with a bounded Brent solve
    between the neighbours of the chosen grid point (``refine=False``
    returns the raw grid point). The exhaustive scan copes with
    multimodal objectives and NaN regions; the refinement recovers the
    precision the grid spacing loses.

    Returns ``best_alpha = NaN`` when the objective is NaN over the whole
    grid; the rules treat this as no selection at this iteration.
    """
    alpha_grid = np.geomspace(bounds[0], bounds[1], num_points)
    func_grid = np.array([func(a) for a in alpha_grid])

    if not np.any(np.isfinite(func_grid)):
        return GridSearchResult(np.nan, func_grid, alpha_grid)

    index = int(np.nanargmin(func_grid))
    alpha = (refine_bracket(func, alpha_grid, index) if refine
             else float(alpha_grid[index]))
    return GridSearchResult(alpha, func_grid, alpha_grid, best_index=index)


def projected_residual_norm_sq(regalpha, state, b_norm):
    r"""
    Computes the squared projected residual norm, using the Tikhonov
    residual filter factors :math:`\alpha^2 / (s^2 + \alpha^2)`.
    """
    reg_sq = regalpha**2
    fr = reg_sq / (state.singular_values_squared + reg_sq)
    residual_k = np.sum(np.square(fr * (b_norm * state.u1)))
    residual_tail = np.square(b_norm * state.u1_tail)
    return float(residual_k + residual_tail)


def _geometry_size(geometry):
    """Total number of elements in a geometry, block-aware."""
    if hasattr(geometry, 'geometries'):
        return sum(_geometry_size(g) for g in geometry.geometries)
    size = 1
    for length in geometry.shape:
        size *= length
    return size

class HybridRule(ABC):
    r"""
    Base class for rules that choose the regularisation parameter of a
    hybrid LSQR run.

    A rule is passed to LSQR as ``alpha``. During an unregularised
    exploratory pass, started from zero, the solver calls :meth:`seed` once
    and :meth:`step` every iteration; the rule re-selects the parameter on
    the projected problem :math:`B_k` and records it in `selected_alpha`.
    :meth:`LSQR.solve_at` then solves the regularised problem at that value.
    The regularisation parameter is called ``regalpha``; ``alpha`` and
    ``beta`` are the Golub-Kahan bidiagonalisation scalars.

    Parameters
    ----------
    tol : float, default 1e-3
        Relative saturation tolerance on successive ``regalpha`` selections.
    mode : {'stop', 'match', 'select_only'}, default 'stop'
        With ``'stop'`` the exploratory pass ends once the selection
        saturates and the realisation runs the full budget of ``run()``.
        With ``'match'`` the realisation runs only the iterations the
        exploratory pass used. With ``'select_only'`` the pass is never
        stopped early, so both passes run the full budget.
    """

    EPS = 1e-12
    # No saturation verdict before this many iterations.
    MIN_ITERATIONS = 10
    GRID_POINTS = 200
    # Selections above this fraction of sigma_max(B_k) lie in the part
    # of the spectrum the GKB window has not resolved yet.
    UNRESOLVED_TOP_FRACTION = 0.1
    # The window counts as settled once sigma_min(B_k) has dropped less
    # than this (relative) over WINDOW_STABLE_LOOKBACK iterations.
    WINDOW_STABLE_TOL = 0.01
    WINDOW_STABLE_LOOKBACK = 10

    def __init__(self, tol=1e-3, mode='stop'):
        if tol <= 0:
            raise ValueError(f"Tolerance must be positive, got {tol}.")
        if mode not in ('stop', 'match', 'select_only'):
            raise ValueError(
                "mode should be chosen from 'stop', 'match' or "
                f"'select_only', got '{mode}'.")
        self.tol = tol
        self.mode = mode
        self.history = IterationHistory()
        # Superseded per-pass histories, archived at each re-seed.
        self.pass_histories = []
        self.converged = False
        self.b_norm = 0.0
        # The current best regularisation parameter; NaN before any
        # selection.
        self.selected_alpha = np.nan
        self.m = None
        self.n = None
        # sigma_min(B_k) per iteration, see _window_has_settled
        self._window_lows = []
        self._last_upper_bound = np.inf

    def step(self, algorithm):
        """
        Advance the rule by one iteration; the solver calls this at the
        end of each unregularised iteration. Raises ``StopIteration``
        once the selection saturates, unless ``mode='select_only'``.
        """
        self.update(algorithm.alpha, algorithm.beta)
        if self.converged and self.mode != 'select_only':
            raise StopIteration

    def seed(self, algorithm=None, initial_alpha=None, initial_beta=None,
             data_size=None, domain_size=None):
        """
        Reset the rule for a new pass, from `algorithm` or from the given
        initial GKB scalars and sizes.
        """
        if algorithm is not None:
            initial_alpha = algorithm.alpha
            initial_beta = algorithm.beta
            data_size = _geometry_size(algorithm.operator.range_geometry())
            domain_size = _geometry_size(algorithm.operator.domain_geometry())
        if len(self.history.gkb_alphas) > 1:
            # Archive the previous pass's history before discarding it.
            self.pass_histories.append(self.history)
        self.history = IterationHistory()
        self.history.gkb_alphas.append(initial_alpha)
        self.history.gkb_betas.append(initial_beta)
        self.b_norm = initial_beta
        self.selected_alpha = np.nan
        self.converged = False
        self.m = data_size
        self.n = domain_size
        self._window_lows = []
        self._last_upper_bound = np.inf

    def return_krylov_state(self):
        """Returns a KrylovState built from the current history."""
        return KrylovState(bk=build_bidiagonal(self.history.gkb_alphas,
                                               self.history.gkb_betas))

    def update(self, alpha, beta):
        """
        Advance one iteration: record the scalar pair, re-select the
        parameter and check saturation.
        """

        self.history.gkb_alphas.append(alpha)
        self.history.gkb_betas.append(beta)
        state = self.return_krylov_state()
        self._window_lows.append(state.min_singular_value)

        upper_bound = state.max_singular_value
        self._last_upper_bound = upper_bound

        new_regalpha = self._calculate_optimal_regalpha(state, upper_bound)

        # A rule that declines to select returns NaN; selected_alpha
        # keeps the last accepted value.
        if np.isfinite(new_regalpha):
            self.selected_alpha = new_regalpha
        self.history.regalphas.append(new_regalpha)

        # The resolution gate is applied here so that subclass convergence
        # shortcuts are subject to it too.
        self.converged = (self._check_convergence()
                          and self._selection_is_resolved(state))

        log.debug("%s iteration %d: regalpha = %.4e", type(self).__name__,
                  state.iteration, self.selected_alpha)

    def _pick_is_assessable(self):
        """
        Preconditions shared by the saturation checks: at least
        MIN_ITERATIONS selections, and a finite selection away from the
        top of the search range, where a pick tracks the unresolved
        subspace rather than the problem.
        """
        if len(self.history.regalphas) < self.MIN_ITERATIONS:
            return False
        return self.selected_alpha < 0.95 * self._last_upper_bound

    def _check_convergence(self):
        """Checks if the regularisation parameter has saturated based on tolerance."""
        if not self._pick_is_assessable():
            return False

        prev = self.history.regalphas[-2]
        denom = abs(self.selected_alpha) + self.EPS

        return (abs(self.selected_alpha - prev) / denom) < self.tol

    def _selection_is_resolved(self, state, pick=None):
        """
        Whether `pick` (default: the selection) lies below the unresolved
        top of the spectrum of B_k, or the window has settled.
        """
        if pick is None:
            pick = self.selected_alpha
        if not np.isfinite(pick):
            # Saturation cannot fire on a NaN selection.
            return True

        smax = state.max_singular_value
        smin = state.min_singular_value
        threshold = smax * self.UNRESOLVED_TOP_FRACTION
        if smin > 0 and smax > 0:
            threshold = min(threshold, float(np.sqrt(smin * smax)))

        if pick < threshold:
            return True
        return self._window_has_settled()

    def _window_has_settled(self):
        """
        True once sigma_min(B_k) has stopped falling, i.e. its relative
        drop over the last WINDOW_STABLE_LOOKBACK iterations is below
        WINDOW_STABLE_TOL. With fewer iterations than the lookback the
        window is treated as still expanding.
        """
        lookback = self.WINDOW_STABLE_LOOKBACK
        lows = self._window_lows
        if len(lows) <= lookback:
            return False
        past = lows[-1 - lookback]
        if past <= 0:
            return True
        return lows[-1] >= (1.0 - self.WINDOW_STABLE_TOL) * past

    def _calculate_optimal_regalpha(self, state, upper_bound):
        """
        Default selection: a grid scan with bracketed log-space refinement
        of :meth:`evaluate_objective`. A minimal rule implements only
        ``evaluate_objective``; a rule with its own search overrides this.
        """
        def func(regalpha):
            return self.evaluate_objective(regalpha, state)

        result = find_optimal_alpha_via_grid(
            func,
            bounds=(self.EPS, upper_bound),
            num_points=self.GRID_POINTS)
        return float(result.best_alpha)

    @abstractmethod
    def evaluate_objective(self, regalpha, state):
        """Evaluates the objective function for a given regalpha."""
