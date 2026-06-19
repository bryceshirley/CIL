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

from cil.framework import DataContainer
from cil.optimisation.algorithms import Algorithm

import numpy as np
import logging

log = logging.getLogger(__name__)


class IRLS(Algorithm):
    r"""
    Iteratively Reweighted Least Squares (IRLS) algorithm for solving L1-regularised problems.
    
    This outer algorithm acts as a meta-solver. It manages an inner Krylov subspace 
    solver (e.g., LSQR or CGLS), iteratively updating a diagonal weight matrix to 
    approximate the L1 norm:

    .. math::
        w_k = (|L u_{k-1}|^2 + \tau_k^2)^{-1/4}

    References
    ----------
    .. [1] R. Chartrand and Wotao Yin, "Iteratively reweighted algorithms for
       compressive sensing," 2008 IEEE ICASSP, Las Vegas, NV, USA, 2008.
    """

    def __init__(
        self,
        inner_solver: Algorithm,
        tau: float = 1.0,
        tau_factor: float = 0.1,
        max_inner_iterations: int = 50,
        reset_state: bool = True,
        **kwargs,
    ):
        """
        Initialise the IRLS algorithm.

        Parameters
        ----------
        inner_solver : Algorithm
            The underlying least squares algorithm to use (e.g., LSQR, CGLS).
        tau : float, optional
            Small positive parameter for L1 regularisation to prevent singularities.
        tau_factor : float, optional
            Factor to decrease tau at each outer iteration. Default is 0.1.
        max_inner_iterations : int, optional
            Maximum number of inner iterations for the least squares algorithm. Default is 50.
        reset_state : bool, optional
            If True, reset the state of the inner solver at each outer iteration. Default is True.
        """
        super().__init__(**kwargs)

        self.inner_solver = inner_solver
        self.max_inner_iterations = max_inner_iterations
        self.reset_state = reset_state
        self.tau = tau
        self.tau_factor = tau_factor

        # Duck typing: Check if the inner solver supports weighted operators
        if not hasattr(self.inner_solver, 'operator') or not hasattr(self.inner_solver.operator, 'weights'):
            raise ValueError("The operator of the inner solver must expose a 'weights' setter for IRLS.")
            
        self.configured = True

    def update(self):
        """Perform a single outer IRLS iteration."""
        
        # Calculate and inject new L1 weights based on the current solution
        self._update_weights()

        # Set up the inner solver for the new run
        if not self.reset_state:
            self.inner_solver.initial = self.inner_solver.get_output()
        
        # Reset the inner solver
        self.inner_solver.reset_state()

        # Inner Loop: Run the Krylov solver
        self.inner_solver.run(self.max_inner_iterations)


    def _update_weights(self):
        """
        Calculates and updates the diagonal weight matrix for L1 regularisation.
        """
        # Pointer to weights
        d = self.inner_solver.operator.weights

        # Replace weights with current solution
        d.fill(self.inner_solver.x)

        # Calculate new L1 weights: w = (Lx^2 + tau^2)^{-1/4}
        d.power(2, out=d)
        d.add(self.tau**2, out=d)
        d.power(-0.25, out=d)

        # Adapt tau for the next outer iteration
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
        log.debug("Tau adapted to: %e", self.tau)

    def get_output(self):
        """Returns the final physical solution from the inner solver."""
        return self.inner_solver.get_output()