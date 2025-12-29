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
        \min_x \| Ax - b \|_2^2 + \alpha^2 \| L x \|_2^2

    where :math:`A` is a linear operator, :math:`b` is the acquired data, :math:`L` is the regularisation operator, and :math:`\alpha` is the regularisation parameter.
    The regularisation parameter is selected at each iteration using a specified rule.

    Parameters
        ----------
        initial : DataContainer, optional
            Initial guess for the solution.
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data.
        regalpha : float, optional
            Regularisation parameter. Default is 0 (no regularisation).
        reg_norm_type : str, optional
            Type of regularisation norm ('L1' or 'L2'). Default is 'L2'.
        weight_operator : Operator, optional
            Regularisation operator :math:`L`. The operator must have an inverse or pseudo-inverse
            method implemented. If no weight_operator is provided, defaults to IdentityOperator
            for 'L2' norm and DiagonalOperator for 'L1' norm.
        maxoutit : int, optional
            Maximum number of outer iterations. Default is the size of the domain.
        maxinit : int, optional
            Maximum number of inner iterations for L1 regularisation.
        tau : float, optional
            Small positive parameter for L1 regularisation.
        atol : float, optional
            Absolute tolerance for stopping criteria for L1 regularisation.
        btol : float, optional
            Relative tolerance for stopping criteria for L1 regularisation.
        xtol : float, optional
            Solution change tolerance for stopping criteria for L1 regularisation.
        hybrid_reg_rule : UpdateRegGCV, UpdateRegDiscrep, UpdateRegLcurve, optional
            Instance of a hybrid regularisation parameter selection rule. If None, defaults to UpdateRegGCV.
    """

    def __init__(
        self,
        operator,
        data: DataContainer,
        initial: DataContainer = None,
        reg_norm_type: str = "L2",
        struct_operator=None,
        regalpha: float = 0.0,
        maxoutit: int = 50,
        maxinit: int = 20,
        tau: float = 1e-3,
        atol: float = 1e-3,
        btol: float = 1e-3,
        xtol: float = 1e-3,
        reinitialize_GKB: bool = True,
        hybrid_reg_rule=None,
        **kwargs,
    ):
        """
        Initialisation of the Hybrid GLSQR algorithm.

        Parameters
        ----------
        initial : DataContainer, optional
            Initial guess for the solution.
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data.
        regalpha : float, optional
            Regularisation parameter. Default is 0 (no regularisation).
        reg_norm_type : str, optional
            Type of regularisation norm ('L1' or 'L2'). Default is 'L2'.
        weight_operator : Operator, optional
            Regularisation operator :math:`L`. The operator must have an inverse or pseudo-inverse
            method implemented. If no weight_operator is provided, defaults to IdentityOperator
            for 'L2' norm and DiagonalOperator for 'L1' norm.
        maxoutit : int, optional
            Maximum number of outer iterations. Default is the size of the domain.
        maxinit : int, optional
            Maximum number of inner iterations for L1 regularisation.
        tau : float, optional
            Small positive parameter for L1 regularisation.
        atol : float, optional
            Absolute tolerance for stopping criteria for L1 regularisation.
        btol : float, optional
            Relative tolerance for stopping criteria for L1 regularisation.
        xtol : float, optional
            Solution change tolerance for stopping criteria for L1 regularisation.
        hybrid_reg_rule : UpdateRegGCV, UpdateRegDiscrep, UpdateRegLcurve, optional
            Instance of a hybrid regularisation parameter selection rule. If None, defaults to UpdateRegGCV.
        """
        # Initialise parent GLSQR class
        super().__init__(operator=operator, data=data, initial=initial,
                         reg_norm_type=reg_norm_type,
                         struct_operator=struct_operator,
                         regalpha=regalpha,
                         maxinit=maxinit,
                         tau=tau,
                         atol=atol,
                         btol=btol,
                         xtol=xtol,
                         maxoutit=maxoutit,
                         reinitialize_GKB = reinitialize_GKB,
                         **kwargs)
        
        
        # Set up hybrid regularisation parameter selection rule
        self.setup_hybridLSQR(hybrid_reg_rule=hybrid_reg_rule)
       
    
    def setup_hybridLSQR(self,hybrid_reg_rule=None):
        """Set up the regularisation parameter selection rule."""

        # Initialise storage for alpha and beta history
        self.alphavec = [self.alpha]
        self.betavec = [self.beta]

        self.k = 1  # Iteration counter for hybrid LSQR

        # Select rule instance
        if hybrid_reg_rule is not None:
            self.reg_rule = hybrid_reg_rule
        else:
            self.reg_rule = UpdateRegGCV(
                tol=1e-3,
                data_size=self.data.size,
                domain_size=self.initial.size,
                gcv_weight=1.0,
                adaptive_weight=True,
            )

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
    
    def _append_scalar_history(self):
        """Store history of alpha and beta."""
        self.alphavec.append(self.alpha)
        self.betavec.append(self.beta)
        self.k += 1

    def update(self):
        """single iteration"""
        # Perform LSQR iteration # TODO: L2: doesn't need to update x here.
        self._perform_iteration()

        # Build Bk
        self._append_scalar_history()
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
