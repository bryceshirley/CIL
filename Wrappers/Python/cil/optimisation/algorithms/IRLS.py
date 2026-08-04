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
from cil.optimisation.utilities.callbacks import Callback, InnerCallback, OuterCallback
from typing import Union, List, Optional
from tqdm.auto import tqdm

import numpy as np
import logging
import sys

log = logging.getLogger(__name__)


class IRLS(Algorithm):
    r"""
    Iteratively Reweighted Least Squares (IRLS) algorithm for solving L1-regularised problems.

    This outer algorithm acts as a meta-solver. It manages an inner solver (e.g., 
    LSQR or CGLS), iteratively updating a diagonal weight matrix to approximate 
    the L1 norm

    ||u||_1 ~ \sum_i w_i |u_i|^2

    .. math::
        w_k = (|u_{k-1}|^2 + \tau_k^2)^{-1/4}

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
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.inner_solver = inner_solver
        self.max_inner_iteration = max_inner_iteration
        self.reset_state = reset_state
        self.tau = tau
        self.tau_factor = tau_factor

        if not hasattr(self.inner_solver, "operator") or not hasattr(
            self.inner_solver.operator, "weights"
        ):
            raise ValueError(
                "The operator of the inner solver must expose a 'weights' setter for IRLS."
            )

        self.configured = True

    def run(
        self,
        iterations: int = None,
        callbacks: Optional[List[Callback]] = None,
        verbose: int = 1,
        **kwargs,
    ):
        """
        Overrides the base run method to automatically inject a tqdm progress bar
        for the outer IRLS iterations, hiding the complexity from the user.
        """
        if iterations is None:
            raise ValueError("`run()` missing number of `iterations`")

        if callbacks is None:
            callbacks = []

        with tqdm(
            total=iterations,
            desc="Outer Loop",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ) as outer_pbar:
            outer_cb = OuterCallback(outer_pbar)
            callbacks.append(outer_cb)

            # Pass the call up to the CIL Algorithm base class. Force verbose=0 to hide CIL's logs.
            super().run(iterations, callbacks=callbacks, verbose=0, **kwargs)

    def update(self):
        """Perform a single outer IRLS iteration."""

        # Calculate and inject new L1 weights based on the current solution
        self._update_weights()

        # Set up the inner solver for the new run
        if not self.reset_state:
            self.inner_solver.initial = self.inner_solver.get_output()

        # Reset the inner solver
        self.inner_solver.reset_state()

        # Inner Loop: Run the Krylov solver with a nested tqdm progress bar
        with tqdm(
            total=self.max_inner_iteration,
            desc="Inner Loop",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        ) as inner_pbar:
            inner_cb = InnerCallback(inner_pbar)
            self.inner_solver.run(
                self.max_inner_iteration, callbacks=[inner_cb], verbose=0
            )

    def _update_weights(self):
        """
        Calculates and updates the diagonal weight matrix for L1 regularisation.
        Updates the existing weights container in-place based on the true previous solution.
        """
        d = self.inner_solver.operator.weights
        op = self.inner_solver.operator

        if hasattr(op, "struct_operator"):
            # Map from solution space to structure space (x_0 = L * u_0)
            op.struct_operator.direct(self.inner_solver.solution, out=d)  # type: ignore
        else:
            d.fill(self.inner_solver.solution)

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

    def get_output(self):
        """Returns the final physical solution from the inner solver."""
        return self.inner_solver.get_output()
