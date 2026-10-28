#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
# CIL Developers and contributers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.optimisation.algorithms import LSQR, CGLS, Algorithm
from cil.optimisation.functions import L1Norm
from cil.optimisation.operators import WeightedOperator
from cil.optimisation.utilities.callbacks import (CGLSEarlyStopping,
                                                  IRLSEarlyStopping)
from cil.optimisation.utilities.hybrid import HybridRule

import logging
import numpy as np
import warnings

log = logging.getLogger(__name__)


class IRLS(Algorithm):
    r"""
    Iteratively Reweighted Least Squares (IRLS) algorithm for solving L1-regularised problems.

    This outer algorithm manages an inner solver (e.g., LSQR or CGLS), iteratively 
    updating a diagonal weight matrix to approximate the L1 norm

    ||u||_1 ~ \sum_i w_i |u_i|^2

    .. math::
        w_k = (|u_{k-1}|^2 + \tau_k^2)^{-1/4}

    where :math:`\tau_k` is a smoothing parameter that is reduced over iterations [1] to improve convergence.
    
    Allowing the inner solver to solve
    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|Lx\|_1

    by iteratively solving a series of weighted L2 problems

    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|W_k Lx\|_2^2

    Where :math:`W_k` is a diagonal weight matrix that is updated by the IRLS algorithm.


    Parameters
    ----------
    initial : DataContainer, optional
        Starting point of the first inner solve, and the point the first
        weights are computed from. Defaults to zeros.
    operator : Operator
        The forward model :math:`A`.
    data : DataContainer
        The measured data :math:`b`.
    alpha : float or HybridRule
        The regularisation parameter :math:`\alpha`. Must be positive and non-zero. 
        A :class:`~cil.optimisation.utilities.hybrid.HybridRule` instead selects
        :math:`\alpha` each outer iteration, recorded in `selected_alphas`; the
        outer loop then stops when the last three selections agree to the rule's `tol`.
    struct_operator : Operator, optional
        The structural operator :math:`L`. Defaults to ``None``, giving standard L1 regularisation.
    inner_solver : class, default LSQR
        The inner solver class, :class:`LSQR` or :class:`CGLS`. IRLS builds
        and owns the instance itself; pass the class, not an instance.
    form : {'block', 'standard'}, default 'block'
        Which formulation the inner solver iterates on; see the LSQR and CGLS
        documentation. The default block form accepts any :math:`L` and can take any initialisation.
    tau : float, default 1.0
        Initial smoothing parameter of the weight update.
    tau_factor : float, default 0.1
        Multiplicative decrease of ``tau`` per outer iteration.
    tau_min : float, default 1e-8
        Floor for ``tau``.
    max_inner_iteration : int, default 20
        Iterations of the inner solver per outer iteration.
    warm_start : bool, default True
        Start every inner solve from the previous outer iterate. Set False to
        restart every inner solve from scratch. Ignored under a hybrid rule.
    tol : float, optional
        Attach an :class:`IRLSEarlyStopping` callback with this tolerance.

    References
    ----------
    .. [1] R. Chartrand and Wotao Yin, "Iteratively reweighted algorithms for
       compressive sensing," 2008 IEEE ICASSP, Las Vegas, NV, USA, 2008.
    """

    def __init__(self, initial=None, operator=None, data=None, alpha=0,
                 struct_operator=None, inner_solver=LSQR, form='block',
                 tau=1.0, tau_factor=0.1, tau_min=1e-8,
                 max_inner_iteration=20, warm_start=True, tol=None, **kwargs):
        super().__init__(**kwargs)

        self.max_inner_iteration = max_inner_iteration
        self.warm_start = warm_start
        self.tau = tau
        self.tau_factor = tau_factor
        self.tau_min = tau_min
        self.tol = tol

        if operator is None or data is None:
            raise ValueError(
                'You must initialise IRLS with an `operator` and `data`')

        self.set_up(initial=initial, operator=operator, data=data,
                    alpha=alpha, struct_operator=struct_operator,
                    inner_solver=inner_solver, form=form)

    def set_up(self, initial, operator, data, alpha=0, struct_operator=None,
               inner_solver=LSQR, form='block'):
        """
        Set up the IRLS algorithm: build the inner solver and its weights.

        See the class documentation for the parameters.
        """
        log.info("%s setting up", self.__class__.__name__)

        if not isinstance(inner_solver, type):
            raise TypeError(
                "inner_solver must be a solver class, e.g. inner_solver=LSQR "
                "or inner_solver=CGLS; IRLS builds the solver itself. "
                "Got an instance of {}.".format(type(inner_solver).__name__))

        self.hybrid = isinstance(alpha, HybridRule)
        self.selected_alphas = []
        if self.hybrid:
            if not hasattr(inner_solver, 'solve_at'):
                raise ValueError(
                    "A hybrid rule requires an inner solver with solve_at, "
                    "got {}.".format(inner_solver.__name__))
        elif alpha <= 0:
            raise ValueError(
                "The regularisation parameter,  alpha,  must be positive, "
                "got alpha = {}.".format(alpha))

        self.weighted_operator = WeightedOperator(
            operator.domain_geometry(), struct_operator)

        # Set up krylov solver
        self.inner_solver = inner_solver(
            initial=initial, operator=operator, data=data, alpha=alpha,
            struct_operator=self.weighted_operator, form=form)

        if not self.hybrid and self.warm_start \
                and not self.inner_solver.supports_warm_start:
            warnings.warn(
                "{} in standard form cannot warm start: it damps the step "
                "rather than the solution, so resuming from the previous "
                "outer iterate does not solve the regularised problem. "
                "Forcing warm_start=False. Use form='block' to warm "
                "start.".format(type(self.inner_solver).__name__),
                UserWarning, stacklevel=3)
            self.warm_start = False

        self.inner_callbacks = ([CGLSEarlyStopping()]
                                if isinstance(self.inner_solver, CGLS) else [])

        # Solution buffer for standard form's (WL)^-1 mapping; block form uses the live iterate.
        self.tmp_solution = (
            self.inner_solver.solution_geometry().allocate(0)
            if self.inner_solver.standard_form else None)

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    @property
    def weights(self):
        """The mutable diagonal IRLS weights, living in ``Range(L)``."""
        return self.weighted_operator.weights

    def run(self, iterations=None, callbacks=None, verbose=1, **kwargs):
        """
        Run the outer loop. ``tol`` is opt-in; a caller-supplied
        :class:`IRLSEarlyStopping` in ``callbacks`` takes precedence.
        """
        callbacks = [] if callbacks is None else list(callbacks)
        if self.tol is not None and not any(
                isinstance(cb, IRLSEarlyStopping) for cb in callbacks):
            callbacks.append(IRLSEarlyStopping(epsilon=self.tol))
        super().run(iterations, callbacks=callbacks, verbose=verbose,
                    **kwargs)

    def update(self):
        """Perform a single outer IRLS iteration."""

        solution = self.inner_solver.get_output(out=self.tmp_solution)

        if self.hybrid and not self.selected_alphas:
            solution = self.inner_solver.initial
        elif self.warm_start or self.hybrid:
            self.inner_solver.initial = solution

        # Calculate and inject new L1 weights based on that solution
        self._update_weights(solution)

        if self.hybrid:
            self.inner_solver._set_regalpha(0)

        self.inner_solver.initialise_variables()
        self.inner_solver.run(self.max_inner_iteration,
                              callbacks=self.inner_callbacks, verbose=0)

        if self.hybrid:
            self._solve_at_selection()

    def _solve_at_selection(self):
        """Realise the hybrid rule's selection, reusing the previous one if
        the rule made none."""
        rule = self.inner_solver.hybrid_rule
        if not np.isfinite(rule.selected_alpha):
            if not self.selected_alphas:
                raise RuntimeError(
                    "The hybrid rule made no selection in the first outer "
                    "iteration; try increasing max_inner_iteration.")
            rule.selected_alpha = self.selected_alphas[-1]
            log.warning(
                "The hybrid rule made no selection in outer iteration %d; "
                "reusing the previous selection %.4e.", self.iteration,
                rule.selected_alpha)
        self.selected_alphas.append(float(rule.selected_alpha))
        self.inner_solver.solve_at()
        self._check_alpha_saturation()

    def _check_alpha_saturation(self):
        """Stop once the last three selections agree to the rule's `tol`."""
        if len(self.selected_alphas) < 3:
            return
        rule = self.inner_solver.hybrid_rule
        newest = self.selected_alphas[-1]
        scale = abs(newest) + rule.EPS
        if all(abs(a - newest) / scale < rule.tol
               for a in self.selected_alphas[-3:]):
            log.info("Hybrid selections saturated at %.4e", newest)
            raise StopIteration

    def _update_weights(self, solution):
        """Update the diagonal L1 weights in-place from the previous outer
        solution."""
        weights = self.weighted_operator.weights

        self.weighted_operator.operator.direct(solution, out=weights)

        # weights = (|Lu|^2 + tau^2)^(-1/4)
        weights.power(2, out=weights)
        weights.add(self.tau**2, out=weights)
        weights.power(-0.25, out=weights)

        self._adapt_tau()

    def update_objective(self):
        """
        The L1 objective ||Au-b||^2 + alpha^2 ||Lu||_1 at the current
        solution, at the cost of one application of A and of L per calculation. Called every ``update_objective_interval outer iterations. ``.
        """
        u = self.inner_solver.get_output()

        residual = self.inner_solver.operator.operator.direct(u)
        residual.sapyb(1.0, self.inner_solver.data, -1.0, out=residual)
        data_term = residual.squared_norm()

        reg_term = L1Norm()(self.weighted_operator.operator.direct(u))
        self.loss.append(data_term
                         + self.inner_solver.regalpha**2 * reg_term)

    def _adapt_tau(self):
        """
        Adapts the smoothing parameter tau.
        Reduces by tau_factor until it hits the tau_min floor.
        """
        self.tau = max(self.tau * self.tau_factor, self.tau_min)

    def get_output(self, out=None):
        """Returns the final physical solution from the inner solver."""
        return self.inner_solver.get_output(out=out)