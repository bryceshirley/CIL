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
    HybridTikhonovOperator, 
    BlockTikhonovOperator,
    TikhonovOperator
)
from cil.optimisation.utilities.hybrid.BaseHybridRule import BaseHybridRule
import numpy as np
import logging
from typing import Optional
from tqdm.auto import tqdm
import sys
from typing import Optional
from cil.utilities.display import show2D

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

    Hybrid regularisation can be applied by providing a `hybrid_rule` that 
    automatically selects the regularisation parameter :math:`\alpha` at 
    each iteration.

    We can also have L1, L1-struct regularisation via Tikhonov Operators:

     - If `hybrid_rule` is used, `HybridTikhonovOperator` is used to transform 
       the problem into standard form: :math:`K = A (WL)^{-1}`, and the penalty 
       is applied via internal scalar damping.

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
        hybrid_rule: Optional[BaseHybridRule] = None
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
            Non-negative regularisation parameter. If zero, standard LSQR is used.
        struct_operator : Operator, optional
            Structured operator for the regularisation.
        hybrid_rule : BaseHybridRule, optional
            Rule for automatic selection of the regularisation parameter at each iteration.
        """
        super().__init__()

        self.hybrid_rule = hybrid_rule
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

        # 1. Hybrid Rule explicitly requires the Hybrid Operator transformation
        if self.hybrid_rule is not None:
            self.operator = HybridTikhonovOperator(
            operator=operator,
            solution_geometry=operator.domain_geometry(),
            struct_operator=struct_operator
            )

        # 2. User passed a struct_operator without hybrid rule
        elif self.regalpha>0 and struct_operator is not None:
            self.operator = BlockTikhonovOperator(
            operator=operator,
            solution_geometry=operator.domain_geometry(),
            regalpha=self.regalpha,
            struct_operator=struct_operator
            )
        else:
            self.operator = TikhonovOperator(
                operator=operator,
                solution_geometry=operator.domain_geometry(),
            )

        if initial is None:
            initial = operator.domain_geometry().allocate(0)


        self.data = data
        self.initial = initial



        # 2. Augment data vector if using a block operator
        # Kx returns a block vector, so b must be augmented with a zero block for residuals
        if isinstance(self.operator, BlockTikhonovOperator):
            zero_block = self.operator.range_geometry().geometries[1].allocate(0)
            self.b = BlockDataContainer(self.data, zero_block) # TODO: Find out if this creates data and if a zero container should be created.
        else:
            self.b = self.data

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
    
    def _bidiag_update(self, input_vec, target_vec, op_func, shift_vec, scalar, buffer1):
        """
        Performs: target = (Op(input) - scalar * shift) / norm
        buffer1: Used to store Op(input). Must match Range of Op.
        """
        # Apply Operator: buffer1 = Op(input_vec)
        op_func(input_vec, out=buffer1)

        # Combine: buffer1 - scalar * shift_vec -> target_vec
        buffer1.sapyb(1.0, shift_vec, -scalar, out=target_vec)

        norm = target_vec.norm()
        if norm > 0:
            target_vec.divide(norm, out=target_vec)
        return norm

    def reset_state(self):
        """
        Golub-Kahan Bidiagonalisation (GKB) Initialisation
        """

        # 1. Copy initial guess or map it to structure space
        if isinstance(self.operator, HybridTikhonovOperator):
            # Map from solution space to structure space (x_0 = L * u_0)
            self.x = self.operator.reg_operator.direct(self.initial)
        else:
            self.x = self.initial.copy()

        # 2. u = (b - Kx) / beta
        # Explicitly calculates b - Kx safely for both block and standard containers
        self.operator.direct(self.x, out=self.tmp_range_data)
        self.b.sapyb(1.0, self.tmp_range_data, -1.0, out=self.u)  # u = 1.0*b - 1.0*Kx
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

        if self.hybrid_rule:
            # Initialise the history of alpha and beta for hybrid regularisation
            self.hybrid_rule.reset_state(self.alpha, self.beta)

    def update(self):
        """single iteration of GKB"""
        # 1 & 2. Advance the subspace basis
        self._generate_next_basis_vectors()

        # 3. Scalar Update
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

        # 4. Vector Updates (Using stored coefficients)
        # x = x + step_coeff * d
        self.x.sapyb(1.0, self.d, self.step_coeff, out=self.x)
        # d = v - d_update_coeff * d
        self.v.sapyb(1.0, self.d, -self.d_update_coeff, out=self.d)

        if self.hybrid_rule:
            self.hybrid_rule.update(
                    alpha=self.alpha, 
                    beta=self.beta
                )
            
    def _generate_next_basis_vectors(self):
        """Advances the Golub-Kahan bidiagonalisation by one step."""
        # 1. Update u: u = (Kv - alpha*u) / beta
        self.beta = self._bidiag_update(
            self.v, self.u, self.operator.direct, self.u, self.alpha, 
            self.tmp_range_data
        )

        # 2. Update v: v = (K*u - beta*v) / alpha
        self.alpha = self._bidiag_update(
            self.u, self.v, self.operator.adjoint, self.v, self.beta, 
            self.tmp_domain
        )

    def update_objective(self):
        """
        Update the objective function value (residual norm squared).
        """
        if np.isnan(self.normr):
            raise StopIteration()
        self.loss.append(self.normr**2)

        if self.hybrid_rule:
            if self.hybrid_rule.stopping_state.converged:
                log.info(
                    "Hybrid LSQR stopping criterion reached at iteration %d", self.hybrid_rule.stopping_state.iteration
                )
                log.info("Selected regularisation parameter: %e", self.hybrid_rule.stopping_state.regalpha)

                raise StopIteration()

    def get_output(self):
        r"""Returns the physical solution."""

        if self.hybrid_rule:

            # Retrieve converged parameters
            iteration = self.hybrid_rule.stopping_state.iteration
            self.regalpha = self.hybrid_rule.stopping_state.regalpha

            log.info(
                "Hybrid LSQR converged at iteration %d with alpha %e.",
                iteration,
                self.regalpha,
            )

            stored_rule = self.hybrid_rule
            self.hybrid_rule = None

            # Reset and run fixed iterations with the selected alpha
            self.reset_state()

            for _ in tqdm(
                range(iteration),
                desc="Computing Solution",
                leave=False,
                dynamic_ncols=True,
            ):
                self.update()

            # Restore rule for plotting/history tools
            self.hybrid_rule = stored_rule
    
            if isinstance(self.operator, HybridTikhonovOperator):
                return self.operator.reg_operator.inverse(self.x)

        log.info("Returning standard LSQR solution.")
        return self.x
