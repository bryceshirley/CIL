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

from cil.optimisation.operators.Operator import Operator
from cil.framework import DataContainer, BlockDataContainer
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.operators.TikhonovOperator import (
    BlockTikhonovOperator,
    TikhonovOperator,
)
import numpy as np
import logging
from typing import Optional, Union
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


class LSQR(Algorithm):
    r"""
    Least Squares with QR factorisation (LSQR) algorithm.

    The LSQR algorithm is used to solve large-scale linear systems and
    least-squares problems, particularly when the matrix is sparse or implicitly
    defined.

    Solves the standard least-squares problem:

    .. math::

        \min_u \|A u - b\|_2^2 + \alpha^2 \| u\|_2^2,

    where :math:`A` is a linear operator, :math:`b` is the acquired data, and
    :math:`\alpha` is the regularisation parameter.

    We can also have L1, L1-struct regularisation via Tikhonov Operators:

     - Otherwise, block structured regularisation `BlockTikhonovOperator` is used
       where the penalty is natively built into the block matrix:
       :math:`K = \begin{bmatrix} A \\ \alpha WL \end{bmatrix}`.

    where :math:`L` is the structured operator and :math:`W` is the weight operator
    for Iteratively Reweighted Regularisation.

    We can achieve L1 regularisation by passing the instantiated LSQR algorithm
    into the `IRLS` algorithm.

    .. math::
        \min_u \|A u - b\|_2^2 + \alpha^2 \|W L u\|_1,

    is equivalent to solving

    .. math::
        \min_u \|A u - b\|_2^2 + \alpha^2 \|W_k L u\|_2^2,

    where :math:`W_k` is a diagonal weight matrix that is updated at each
    outer iteration.

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/lsqr/
    """

    def __init__(
        self,
        operator,
        data: DataContainer,
        initial: Optional[DataContainer] = None,
        regalpha: float = 0.0,
        struct_operator: Optional[Operator] = None,
    ):
        """
        Initialise the LSQR algorithm.

        Parameters
        ----------
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data (right-hand side of the equation).
        initial : DataContainer, optional
            Initial guess for the solution. If not provided, a zero-initialised container is used.
        regalpha : float, optional
            Non-negative regularisation parameter. 
        struct_operator : Operator, optional
            Structured operator for the regularisation.
        """
        super().__init__()

        # Initialise the algorithm
        self.set_up(
            initial=initial,
            operator=operator,
            data=data,
            struct_operator=struct_operator,
            regalpha=regalpha,
        )

    def set_up(self, initial, operator, data, struct_operator, regalpha):
        """
        Set up the LSQR algorithm with the problem definition and memory buffers.
        """
        log.info("%s setting up", self.__class__.__name__)
        self.regalpha = regalpha

        if self.regalpha > 0:
            if struct_operator is not None:
                self.operator = BlockTikhonovOperator(
                    operator=operator,
                    solution_geometry=operator.domain_geometry(),
                    regalpha=self.regalpha,
                    struct_operator=struct_operator,
                )
                print("Using BlockTikhonovOperator for regularisation.")
            else:
                self.operator = TikhonovOperator(
                    operator=operator,
                    solution_geometry=operator.domain_geometry(),
                )
                print("Using TikhonovOperator for regularisation with change of variable.")
        else:
            self.operator = operator

        if initial is None:
            initial = operator.domain_geometry().allocate(0)

        self.data = data
        self.initial = initial

        # 3. Identify Geometries based on the active operator
        self.domain_geom_solution = self.operator.domain_geometry()
        self.range_geom_data = self.operator.range_geometry()

        # 4. Problem sizes
        self.domain_size = int(np.prod(self.domain_geom_solution.shape))
        self.data_size = int(np.prod(self.range_geom_data.shape))

        # 5. Allocate variables in their correct spaces
        self.x = self.domain_geom_solution.allocate(0)
        self.v = self.domain_geom_solution.allocate(0)
        self.d = self.domain_geom_solution.allocate(0)
        self.u = self.range_geom_data.allocate(0)

        # Temporary Buffers
        self.tmp_range_data = self.range_geom_data.allocate(0)
        self.tmp_domain = self.domain_geom_solution.allocate(0)

        # Initialise Golub-Kahan bidiagonalisation (GKB)
        self.reset_state()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    def reset_state(self):
        """
        Golub-Kahan Bidiagonalisation (GKB) Initialisation
        """

        # 1. Map initial guess to the correct search space
        if isinstance(self.operator, TikhonovOperator):
            # Standard form operates in structure space: x_0 = W * L * u_0
            self.operator.reg_operator.direct(self.initial, out=self.x)
        else:
            # Block form operates in the physical domain: x_0 = u_0
            self.x = self.initial.copy()

        # 2. u = (b - Kx) / beta
        # Calculate Kx and store it temporarily in u
        self.operator.direct(self.x, out=self.u)
        
        # Perform b - Kx block-wise to avoid allocating a zero container
        if isinstance(self.operator, BlockTikhonovOperator):
            # Block 0: u[0] = 1.0 * b - 1.0 * Kx[0]
            self.data.sapyb(1.0, self.u[0], -1.0, out=self.u[0])
            # Block 1: u[1] = 0.0 - 1.0 * Kx[1]
            self.u[1].multiply(-1.0, out=self.u[1])
        else:
            # Standard single container: u = 1.0 * b - 1.0 * Kx
            self.data.sapyb(1.0, self.u, -1.0, out=self.u)
            
        self.beta = self.u.norm()
        if self.beta > 0:
            self.u.divide(self.beta, out=self.u)

        # 3. v = K*u / alpha
        self.operator.adjoint(self.u, out=self.v)
        self.alpha = self.v.norm()
        if self.alpha > 0:
            self.v.divide(self.alpha, out=self.v)

        # 4. Initialize scalars and search direction
        self.rhobar, self.phibar = self.alpha, self.beta
        self.normr = self.beta
        self.beta0 = self.beta
        self.res2 = 0.0
        self.d = self.v.copy()

    def update(self):
        """Single iteration of the LSQR algorithm (GKB and vector updates)."""
        
        # ---------------------------------------------------------
        # 1 & 2: Advance Golub-Kahan Bidiagonalisation
        # ---------------------------------------------------------
        # Update u: u = (Kv - alpha*u) / beta
        self.operator.direct(self.v, out=self.tmp_range_data)
        self.tmp_range_data.sapyb(1.0, self.u, -self.alpha, out=self.u)
        
        self.beta = self.u.norm()
        if self.beta > 0:
            self.u.divide(self.beta, out=self.u)

        # Update v: v = (K*u - beta*v) / alpha
        self.operator.adjoint(self.u, out=self.tmp_domain)
        self.tmp_domain.sapyb(1.0, self.v, -self.beta, out=self.v)
        
        self.alpha = self.v.norm()
        if self.alpha > 0:
            self.v.divide(self.alpha, out=self.v)

        # ---------------------------------------------------------
        # 3: Scalar Updates
        # ---------------------------------------------------------
        if isinstance(self.operator, BlockTikhonovOperator):
            # No damping needed from LSQR itself
            rhobar1 = self.rhobar
            psi = 0.0
            phibar_temp = self.phibar
        else:
            # Standard LSQR damping logic
            rhobar1 = np.hypot(self.rhobar, self.regalpha)
            psi = (self.regalpha / rhobar1) * self.phibar
            phibar_temp = (self.rhobar / rhobar1) * self.phibar

        rho = np.hypot(rhobar1, self.beta)
        c, s = rhobar1 / rho, self.beta / rho

        # Store coefficients for stopping criteria and updates
        self.step_coeff = (c * phibar_temp) / rho
        self.d_update_coeff = (s * self.alpha) / rho

        # Update class state for next iteration
        self.rhobar = -c * self.alpha
        self.phibar = s * phibar_temp
        self.res2 += psi**2
        self.normr = np.hypot(self.phibar, np.sqrt(self.res2))

        # ---------------------------------------------------------
        # 4: Vector Updates
        # ---------------------------------------------------------
        # x = x + step_coeff * d
        self.x.sapyb(1.0, self.d, self.step_coeff, out=self.x)
        # d = v - d_update_coeff * d
        self.v.sapyb(1.0, self.d, -self.d_update_coeff, out=self.d)

    def update_objective(self):
        """
        Update the objective function value (residual norm squared).
        """
        if np.isnan(self.normr):
            raise StopIteration()
        self.loss.append(self.normr**2)

    def get_output(self):
        r"""Returns the physical solution."""

        if isinstance(self.operator, TikhonovOperator):
            return self.operator.reg_operator.inverse(self.x)

        return self.x