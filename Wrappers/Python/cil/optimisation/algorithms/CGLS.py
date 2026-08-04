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

from cil.framework import BlockDataContainer
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.operators.TikhonovOperator import (
    BlockTikhonovOperator,
    TikhonovOperator,
)
import numpy
import logging
import warnings

log = logging.getLogger(__name__)


class CGLS(Algorithm):
    r"""Conjugate Gradient Least Squares (CGLS) algorithm

    The Conjugate Gradient Least Squares (CGLS) algorithm is commonly used for solving
    large systems of linear equations, due to its fast convergence.

    Problem:

    .. math::
      \min_u \| A u - b \|^2_2

    If a positive regularisation parameter :math:`\alpha` is provided, this algorithm
    uses a `BlockTikhonovOperator` to solve the regularised problem

    .. math::
      \min_u \| A u - b \|^2_2 + \alpha^2 \| W L u \|^2_2

    This is equivalent to the standard CGLS problem

    .. math::
      \min_u \| K u - \hat{b} \|^2

    where

    .. math::

      K = \begin{bmatrix} A \\ \alpha W L \end{bmatrix}, \quad \hat{b} = \begin{bmatrix} b \\ 0 \end{bmatrix}

    where :math:`L` is the structured operator and :math:`W` is the weight operator
    for Iteratively Reweighted Least Squares (IRLS).

    If no structured operator is provided (L = I), a standard `TikhonovOperator` is
    used and the L2 penalty is solved efficiently using native scalar updates.

    Parameters
    ------------
    operator : Operator
        Linear operator for the inverse problem (:math:`A`).
    data : DataContainer
        Acquired data to reconstruct (:math:`b`).
    initial : DataContainer, optional
        Initial guess in the domain of the operator. Default is a container filled with zeros.
    regalpha : float, optional
        Non-negative regularisation parameter (:math:`\alpha`). If zero, standard CGLS is used.
    struct_operator : Operator, optional
        Structured operator for the regularisation (:math:`L`). Default is None.

    Note
    -----
    Passing tolerance directly to CGLS is being deprecated. Instead we recommend using
    the callback functionality: https://tomographicimaging.github.io/CIL/nightly/optimisation/#callbacks
    and in particular the CGLSEarlyStopping callback replicates the old behaviour.

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/cgls/
    """

    def __init__(
        self,
        initial=None,
        operator=None,
        data=None,
        regalpha=0.0,
        struct_operator=None,
        **kwargs,
    ):
        """initialisation of the algorithm"""
        # We are deprecating tolerance
        self.tolerance = kwargs.pop("tolerance", None)
        if self.tolerance is not None:
            warnings.warn(
                stacklevel=2,
                category=DeprecationWarning,
                message="Passing tolerance directly to CGLS is being deprecated. Instead we recommend using the callback functionality: https://tomographicimaging.github.io/CIL/nightly/optimisation/#callbacks and in particular the CGLSEarlyStopping callback replicated the old behaviour",
            )
        else:
            self.tolerance = 0

        super(CGLS, self).__init__(**kwargs)

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)

        if initial is not None and operator is not None and data is not None:
            self.set_up(
                initial=initial,
                operator=operator,
                data=data,
                regalpha=regalpha,
                struct_operator=struct_operator,
            )

    def set_up(self, initial, operator, data, regalpha=0.0, struct_operator=None):
        r"""Initialisation of the algorithm and internal buffers"""

        log.info("%s setting up", self.__class__.__name__)

        self.initial = initial
        self.data = data
        self.regalpha = regalpha

        # 1. Setup the mathematical operator exactly like LSQR
        if self.regalpha > 0:
            if struct_operator is not None:
                self.operator = BlockTikhonovOperator(
                    operator=operator,
                    solution_geometry=operator.domain_geometry(),
                    regalpha=self.regalpha,
                    struct_operator=struct_operator,
            )
            else:
                self.operator = TikhonovOperator(
                    operator=operator,
                    solution_geometry=operator.domain_geometry(),
                )
        else:
            self.operator = operator

        # 2. Allocate persistent buffers (no augmented zero container needed)
        self.q = self.operator.range_geometry().allocate()
        self.r = self.operator.range_geometry().allocate(0)
        self.s = self.operator.domain_geometry().allocate(0)

        # 3. Initialize state
        self.reset_state()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    def reset_state(self):
        """
        Resets the Krylov subspace and residuals to start a new CGLS run.
        """
        # 1. Map initial guess to the correct search space
        if isinstance(self.operator, TikhonovOperator):
            # Standard form operates in structure space: x_0 = W * L * u_0
            # direct() will automatically allocate the correct geometry for self.x
            self.x = self.operator.reg_operator.direct(self.initial)
        else:
            # Block form operates in the physical domain: x_0 = u_0
            self.x = self.initial.copy()

        # Calculate initial residual r = b - Kx efficiently
        self.operator.direct(self.x, out=self.r)
        
        if isinstance(self.operator, BlockTikhonovOperator):
            self.data.sapyb(1.0, self.r[0], -1.0, out=self.r[0])
            self.r[1].multiply(-1.0, out=self.r[1])
        else:
            self.data.sapyb(1.0, self.r, -1.0, out=self.r)

        # Normal equations residual: s = K^T r
        self.operator.adjoint(self.r, out=self.s)
        
        # Add scalar penalty s = s - alpha^2 x natively for standard Tikhonov
        if isinstance(self.operator, TikhonovOperator) and self.regalpha > 0:
            self.s.sapyb(1.0, self.x, -(self.regalpha**2), out=self.s)

        self.p = self.s.copy()

        self.norms0 = self.s.norm()
        self.norms = self.norms0

        self.gamma = self.norms0**2
        self.normx = self.x.norm()

    def update(self):
        """single iteration of CGLS"""

        self.operator.direct(self.p, out=self.q)
        delta = self.q.squared_norm()
        
        # Include scalar regularization natively
        if isinstance(self.operator, TikhonovOperator) and self.regalpha > 0:
            delta += (self.regalpha**2) * self.p.squared_norm()
            
        alpha = self.gamma / delta

        self.x.sapyb(1, self.p, alpha, out=self.x)
        self.r.sapyb(1, self.q, -alpha, out=self.r)

        self.operator.adjoint(self.r, out=self.s)
        
        # Include scalar regularization natively
        if isinstance(self.operator, TikhonovOperator) and self.regalpha > 0:
            self.s.sapyb(1.0, self.x, -(self.regalpha**2), out=self.s)

        self.norms = self.s.norm()
        self.gamma1 = self.gamma
        self.gamma = self.norms**2
        self.beta = self.gamma / self.gamma1
        self.p.sapyb(self.beta, self.s, 1, out=self.p)

        self.normx = (
            self.x.norm()
        )  # TODO: Deprecated, remove when CGLS tolerance is removed

    def update_objective(self):
        a = self.r.squared_norm()
        
        if isinstance(self.operator, TikhonovOperator) and self.regalpha > 0:
            a += (self.regalpha**2) * self.x.squared_norm()
            
        if a is numpy.nan:
            raise StopIteration()
        self.loss.append(a)

    def should_stop(self):  # TODO: Deprecated, remove when CGLS tolerance is removed
        return self.flag() or super().should_stop()

    def flag(self):  # TODO: Deprecated, remove when CGLS tolerance is removed
        """returns whether the tolerance has been reached"""
        flag = (self.norms <= self.norms0 * self.tolerance) or (
            self.normx * self.tolerance >= 1
        )

        if flag:
            self.update_objective()
            log.info("Tolerance is reached: {}".format(self.tolerance))

        return flag

    def get_output(self):
        """Returns the current physical solution"""
        if isinstance(self.operator, TikhonovOperator):
            return self.operator.reg_operator.inverse(self.x)
        return self.x