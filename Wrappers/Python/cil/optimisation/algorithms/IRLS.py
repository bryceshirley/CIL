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
from cil.optimisation.utilities.callbacks import (Callback, CGLSEarlyStopping,
                                                  InnerCallback,
                                                  IRLSEarlyStopping,
                                                  OuterCallback,
                                                  ProgressCallback)
from typing import Union, List, Optional
from tqdm.auto import tqdm

import numpy as np
import logging
import sys
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

    Choosing an inner solver
    ------------------------
    Either :class:`LSQR` or :class:`CGLS` will do, but they behave differently
    as the reweighting proceeds, and the differences are not cosmetic.

    * **Conditioning.** The weights grow like
      :math:`(|Lu|^2 + \tau^2)^{-1/4}` and :math:`\tau` shrinks every outer
      iteration, so the inner problem gets steadily worse conditioned. LSQR
      works with the condition number of :math:`K`; CGLS works with its
      square, and in float32 that is enough to lose conjugacy and diverge.
      IRLS therefore attaches a :class:`CGLSEarlyStopping` guard to a CGLS
      inner solver by default, which restores agreement with LSQR to five
      figures. Override with ``inner_callbacks``.

    * **Warm starts.** LSQR must be used with ``form='block'`` if the previous
      outer iterate is to be reused: in standard form it eliminates
      :math:`\alpha` with a Givens rotation applied to the bidiagonalisation
      of the residual of :math:`x_0`, so the penalty lands on the step rather
      than the solution. CGLS subtracts the :math:`\alpha^2 x` term explicitly
      using the whole iterate and warm starts correctly in either form. IRLS
      checks ``inner_solver.supports_warm_start`` and forces ``reset_state``
      when it has to.

    Stopping the outer loop
    -----------------------
    By default the outer loop runs exactly the number of iterations passed to
    :meth:`run`. Passing ``tol`` instead attaches an :class:`IRLSEarlyStopping`
    callback, which terminates once the relative change between successive
    outer iterates falls below it. Watch the iterate, not the objective:
    :meth:`update_objective` records the inner solver's residual against a
    different reweighted operator each outer iteration, so it is not a fixed
    quantity being minimised and plateaus while the iterates are still moving.

    References
    ----------
    .. [1] R. Chartrand and Wotao Yin, "Iteratively reweighted algorithms for
       compressive sensing," 2008 IEEE ICASSP, Las Vegas, NV, USA, 2008.
    """

    def __init__(
        self,
        inner_solver: Union[LSQR, CGLS],
        tau: float = 1.0,
        tau_factor: float = 0.1,
        max_inner_iteration: int = 20,
        reset_state: bool = False,
        inner_callbacks: Optional[List[Callback]] = None,
        tol: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.inner_solver = inner_solver
        self.max_inner_iteration = max_inner_iteration
        self.reset_state = reset_state
        self.tau = tau
        self.tau_factor = tau_factor
        self.tol = tol

        # Reweighting makes the inner problem progressively worse conditioned:
        # the weights go like (|Lu|^2 + tau^2)^(-1/4) and tau falls by
        # tau_factor every outer iteration, so max(w)/min(w) grows without
        # bound. CGLS runs CG on the explicitly formed normal equations and
        # picks up the square of that condition number, which in float32 is
        # enough to lose conjugacy and diverge: measured on a 128^2 cone-beam
        # problem, ||u|| reached 1.3e+05 at the third outer iteration before
        # falling back. LSQR eliminates each row with a Givens rotation and is
        # unaffected, which is why the guard is only attached to CGLS.
        #
        # CGLSEarlyStopping ends an inner solve once it has converged, or once
        # the iterate runs away. With it, CGLS reproduces LSQR's trajectory to
        # five figures. Pass `inner_callbacks=[]` to opt out.
        if inner_callbacks is None:
            inner_callbacks = ([CGLSEarlyStopping()]
                               if isinstance(inner_solver, CGLS) else [])
        self.inner_callbacks = list(inner_callbacks)

        if not hasattr(self.inner_solver, "operator") or not hasattr(
            self.inner_solver.operator, "weights"
        ):
            raise ValueError(
                "The operator of the inner solver must expose a 'weights' setter for IRLS."
            )

        # The weights are the one container IRLS is entitled to create outside
        # the inner solver's set_up. Prefer that the caller passed weighted=True,
        # in which case this is a no-op and the whole budget stays in set_up.
        if self.inner_solver.weights is None:
            log.info("Allocating IRLS weights. Construct the inner solver with "
                     "weighted=True to keep every allocation inside set_up.")
            self.inner_solver.enable_weights()

        if not self.inner_solver.supports_warm_start and not self.reset_state:
            warnings.warn(
                "{} in standard form cannot warm start: it damps the step "
                "rather than the solution, so resuming from the previous outer "
                "iterate does not solve the regularised problem. Forcing "
                "reset_state=True. Build the inner solver with form='block' to "
                "warm start.".format(type(self.inner_solver).__name__),
                UserWarning, stacklevel=2)
            self.reset_state = True

        # Scratch for the physical solution. In standard form get_output() has to
        # map the iterate back through (WL)^-1, which would otherwise allocate on
        # every outer iteration. Note this is the *solution* geometry, which in
        # standard form is not the solver's domain: that one is Range(L).
        self.tmp_solution = (
            self.inner_solver.operator.reg_operator.domain_geometry().allocate(0)
            if self.inner_solver.standard_form else None)

        self.configured = True

    def run(self, iterations=None, callbacks: Optional[List[Callback]] = None, verbose: int = 1):
        """
        Overrides the base run method to automatically inject a tqdm progress bar
        for the outer IRLS iterations, hiding the complexity from the user.
        """
        if iterations is None:
            raise ValueError("`run()` missing number of `iterations`")

        if callbacks is None:
            callbacks = [ProgressCallback(verbose=verbose)]
        else:
            # Do not append the outer-loop callback to the caller's own list.
            callbacks = list(callbacks)

        # `tol` is opt-in, so the default is still to run the outer loop the
        # number of times asked for. A caller who has already put their own
        # IRLSEarlyStopping in `callbacks` gets theirs rather than two.
        if self.tol is not None and not any(
                isinstance(cb, IRLSEarlyStopping) for cb in callbacks):
            callbacks.append(IRLSEarlyStopping(epsilon=self.tol,
                                               verbose=verbose))

        # verbose=0 has to reach the bars too, or a caller silencing the
        # algorithm still gets two of them per outer iteration.
        self._quiet = verbose == 0

        with tqdm(
            total=iterations,
            desc="Outer Loop",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=self._quiet,
        ) as outer_pbar:
            outer_cb = OuterCallback(outer_pbar)
            callbacks.append(outer_cb)

            # Pass the call up to the CIL Algorithm base class. Force verbose=0 to hide CIL's logs.
            super().run(iterations, callbacks=callbacks, verbose=0)

    def update(self):
        """Perform a single outer IRLS iteration."""

        # Snapshot the current solution *before* touching the weights. In
        # standard form get_output() applies (WL)^-1, so it has to be read while
        # W is still the one the iterate was computed under; overwriting the
        # weights first would map x back through the wrong operator.
        solution = self.inner_solver.get_output(out=self.tmp_solution)

        if not self.reset_state:
            self.inner_solver.initial = solution

        # Calculate and inject new L1 weights based on that solution
        self._update_weights(solution)

        # Reset the inner solver
        self.inner_solver.initialise_variables()

        # Inner Loop: Run the Krylov solver with a nested tqdm progress bar
        with tqdm(
            total=self.max_inner_iteration,
            desc="Inner Loop",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=getattr(self, '_quiet', False),
        ) as inner_pbar:
            inner_cb = InnerCallback(inner_pbar)
            self.inner_solver.run(
                self.max_inner_iteration,
                callbacks=[inner_cb] + self.inner_callbacks,
                verbose=0,
            )

    def _update_weights(self, solution):
        """
        Calculates and updates the diagonal weight matrix for L1 regularisation.

        Updates the existing weights container in-place, so this allocates
        nothing.

        Parameters
        ----------
        solution : DataContainer
            The physical solution :math:`u` from the previous outer iteration,
            read before the weights were touched.
        """
        d = self.inner_solver.operator.weights
        op = self.inner_solver.operator

        if hasattr(op, "struct_operator"):
            # Map from solution space to structure space (Lu), which is where
            # the weights live. Note L, not WL: the penalty being approximated
            # is ||Lu||_1.
            op.struct_operator.direct(solution, out=d)  # type: ignore
        else:
            d.fill(solution)

        # d = (|d|^2 + tau^2)^(-1/4)
        d.power(2, out=d)
        d.add(self.tau**2, out=d)
        d.power(-0.25, out=d)

        self._adapt_tau()

    def update_objective(self):
        """
        Update the objective function value.
        Tracks the final residual norm squared from the inner solver's current run.
        """
        if len(self.inner_solver.loss) > 0:
            self.loss.append(self.inner_solver.loss[-1])
        else:
            self.loss.append(np.nan)

    def _adapt_tau(self):
        """
        Adapts the smoothing parameter tau.
        Reduces by tau_factor until it hits a floor of 1e-8.
        """
        self.tau = max(self.tau * self.tau_factor, 1e-8)

    def get_output(self, out=None):
        """Returns the final physical solution from the inner solver."""
        return self.inner_solver.get_output(out=out)