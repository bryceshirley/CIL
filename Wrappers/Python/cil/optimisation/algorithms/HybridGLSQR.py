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
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.framework import DataContainer
from cil.optimisation.algorithms.GLSQR import GLSQR
from cil.optimisation.utilities.HybridUpdateReg import UpdateRegGCV
import numpy as np
import logging

log = logging.getLogger(__name__)


class HybridGLSQR(GLSQR):
    r"""Hybrid Generalized Least Squares QR (GLSQR) algorithm

    Solves the regularised least-squares problem with hybrid regularisation.
    .. math::
        \min_u \| Au - b \|_2^2 + \alpha^2 \| L u \|_2^2

    where :math:`A` is a linear operator, :math:`b` is the acquired data, :math:`L` 
    is the regularisation operator, and :math:`\alpha` is the regularisation parameter.
    The regularisation parameter is selected at each iteration using a specified rule.

    The GLSR algorithm transforms the problem into the standard tikhonov form

    .. math::

        \min_x \|K x - b\|_2^2 + \alpha^2 \| x \|_2^2,

    where :math:`x = L u` and :math:`K = A L^{-1}`.

    The hybrid aspect of the algorithm comes from the automatic selection of the 
    regularisation parameter :math:`\alpha` at each iteration using a specified rule, 
    such as GCV, Reginska, UPRE, Discrepancy Principle, or L-curve.
    """

    def __init__(
        self,
        operator,
        data: DataContainer,
        initial: DataContainer = None,
        reg_norm_type: str = "L2",
        struct_operator=None,
        regalpha: float = 0.0,
        xtol: float = 0.1, # Relative tolerance for normal equations for IRLS inner loop
        tau: float = 1.0,
        tau_factor: float = 0.1,
        reinitialize_GKB: bool = True,
        max_inner_iterations: int = 50, # Default maximum inner iterations for IRLS
        hybrid_reg_rule=None,
        **kwargs,
    ):
        """
        Initialisation of the Hybrid GLSQR algorithm.

        Parameters
        ----------
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data (right-hand side of the equation).
        initial : DataContainer, optional
            Initial guess for the solution. If not provided, a zero-initialised container is used.
        regalpha : float, optional
            Non-negative regularisation parameter. If zero, standard LSQR is used.
        reg_norm_type : str, optional
            Type of regularisation norm ('L1' or 'L2'). Default is 'L2'.
        struct_operator : Operator, optional
            Regularisation operator :math:`L`. If not provided, defaults to IdentityOperator.
        tau : float, optional
            Small positive parameter for L1 regularisation.
        xtol : float, optional
            Relative tolerance for normal equations for IRLS inner loop. Default is 0.1.
        tau_factor : float, optional
            Factor to decrease tau at each outer iteration for L1 regularisation. Default is 0.1.
        reinitialize_GKB : bool, optional
            Whether to reinitialize the Golub-Kahan Bidiagonalisation (GKB) at each outer iteration for L1 regularisation. Default is True.
        max_inner_iterations : int, optional
            Maximum number of inner iterations for IRLS regularisation. Default is 50.
        hybrid_reg_rule : UpdateRegGCV, UpdateRegDiscrep, UpdateRegLcurve, optional
            Instance of a hybrid regularisation parameter selection rule. If None, defaults to UpdateRegGCV.
        """
        # Initialise parent GLSQR class
        super().__init__(operator=operator,
                        data=data,
                        initial=initial,
                        reg_norm_type=reg_norm_type,
                        struct_operator=struct_operator,
                        regalpha=regalpha,
                        xtol=xtol,
                        tau=tau,
                        tau_factor=tau_factor,
                        reinitialize_GKB=reinitialize_GKB,
                        max_inner_iterations=max_inner_iterations,
                         **kwargs)
        
        # Set up hybrid regularisation parameter selection rule
        self.setup_hybridLSQR(hybrid_reg_rule=hybrid_reg_rule)
       
    
    def setup_hybridLSQR(self,hybrid_reg_rule=None):
        """Set up the regularisation parameter selection rule."""

        # Select rule instance
        if hybrid_reg_rule is not None:
            self.reg_rule = hybrid_reg_rule
        else:
            self.reg_rule = UpdateRegGCV(
                tol=1e-3,
                data_size=self.data_size,
                domain_size=self.domain_size,
                gcv_weight=1.0,
                adaptive_weight=True,
            )
    
    def _initialize_GKB(self):
        """
        Override parent method to also initialize subspace history for 
        hybrid regularisation.
        """
        super()._initialize_GKB()

        self._initialize_subspace_history()
    
    def _GKB_step(self):
        """
        Override parent method to perform a GKB step and then update the 
        regularisation parameter using the hybrid rule.
        """
        super()._GKB_step()
        self._update_subspace_history()

    def _initialize_subspace_history(self):
        """Initialise history of alpha and beta."""
        self.alphavec = [self.alpha]
        self.betavec = [self.beta]
        self.k = 1  # Iteration counter for hybrid LSQR
    
    def _update_subspace_history(self):
        """Store history of alpha and beta."""
        self.alphavec.append(self.alpha)
        self.betavec.append(self.beta)
        self.k += 1

    def _build_projected_operator(self):
        """
        Builds the (k+1) x k bidiagonal projected operator Bk.
        """
        # 2. Pre-allocate Bk for this specific subspace size
        Bk = np.zeros((self.k + 1, self.k))

        # 3. Fill main diagonal: alpha_1 to alpha_k
        np.fill_diagonal(Bk, self.alphavec[:self.k])

        # 4. Fill sub-diagonal: beta_2 to beta_k+1
        np.fill_diagonal(Bk[1:, :], self.betavec[1:self.k+1])
        return Bk

    def update(self):
        """Override parent method to perform a single iteration of the GLSQR 
        algorithm with hybrid regularisation."""
        # Perform a single LSQR iteration of GLSQR with optional IRLS for L1.
        super().update()

        # Build Bk
        Bk = self._build_projected_operator()

        # Select regularisation parameter
        self.reg_rule.update_regularizationparam(Bk=Bk, b_norm=self.betavec[0])

        # Update regularisation parameter in solver
        self.regalpha = self.reg_rule.regalpha

    def update_objective(self):
        """Monitor convergence and loss."""
        super().update_objective()

        if self.reg_rule.converged:
            # Sync the solver's regalpha with the rule's current suggestion
            self.regalpha = self.reg_rule.regalpha
            self.iteration = self.reg_rule.iteration
            log.info(
                "Hybrid LSQR stopping criterion reached at iteration %d", self.iteration
            )
            log.info("Selected regularisation parameter: %e", self.regalpha)
            raise StopIteration()
