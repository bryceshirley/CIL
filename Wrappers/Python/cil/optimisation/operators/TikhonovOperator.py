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
# CIL Developers, listed at:
# https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import logging

from cil.optimisation.operators import (
    LinearOperator,
    IdentityOperator,
    BlockOperator,
)
from cil.optimisation.operators.BlockDiagonalOperator import DiagonalOperator
from cil.optimisation.operators.GradientOperator import GradientOperator

log = logging.getLogger(__name__)


class BlockTikhonovOperator(BlockOperator):
    r"""
    Block Tikhonov Operator for general L2 and L1 regularisation.

    This block operator performs analysis regularisation directly in the
    physical solution space ``u``. It represents

    .. math::

        K =
        \begin{bmatrix}
            A \\
            \alpha W L
        \end{bmatrix}

    It should be the default choice for CGLS or standard LSQR when hybrid
    parameter selection is not being used.

    Geometries
    ----------
    Domain:
        Domain(A), the physical solution space.

    Range:
        BlockGeometry(Range(A), Range(L)).
    """

    def __init__(
        self,
        operator,
        solution_geometry,
        struct_operator=None,
        regalpha: float = 0.0,
        tmp_range_struct=None,
    ):
        if struct_operator is None:
            struct_operator = IdentityOperator(solution_geometry)

        self.operator = operator
        self.struct_operator = struct_operator

        self.reg_operator = WeightedStructOperator(
            domain_geometry=solution_geometry,
            struct_operator=struct_operator,
            tmp_range_struct=tmp_range_struct,
        )

        self.scaled_reg_operator = regalpha * self.reg_operator

        super(BlockTikhonovOperator, self).__init__(
            self.operator,
            self.scaled_reg_operator,
        )

    @property
    def regalpha(self):
        """Return the current Tikhonov regularisation parameter."""
        return self.scaled_reg_operator.scalar

    @regalpha.setter
    def regalpha(self, value: float):
        """Dynamically update the regularisation parameter."""
        self.scaled_reg_operator.scalar = value

    @property
    def weights(self):
        """
        Return the mutable diagonal IRLS weights.

        The weights live in Range(L), not necessarily in Domain(A).
        """
        return self.reg_operator.weights


class TikhonovOperator(LinearOperator):
    r"""
    Standard-form transformation operator for Hybrid LSQR.

    This transforms the generalised Tikhonov problem into standard form by
    mapping the LSQR search space from the physical domain into the regularised
    structure space.

    .. math::

        K = A L_{\text{weighted}}^{-1} = A (W L)^{-1}

    The solver operates in the structure space ``x``. Once the inner Krylov
    solver has found ``x``, the physical solution is recovered through

    .. math::

        u = L_{\text{weighted}}^{-1} x.

    Geometries
    ----------
    Domain:
        Range(L), the regularised structure space.

    Range:
        Range(A), the physical data space.
    """

    def __init__(
        self,
        operator,
        solution_geometry,
        struct_operator=None,
        tmp_domain=None,
        tmp_range_struct=None,
    ):
        if struct_operator is None:
            struct_operator = IdentityOperator(solution_geometry)

        self.operator = operator
        self.struct_operator = struct_operator

        self.reg_operator = WeightedStructOperator(
            domain_geometry=solution_geometry,
            struct_operator=struct_operator,
            tmp_range_struct=tmp_range_struct,
        )

        hybrid_domain = struct_operator.range_geometry()
        hybrid_range = operator.range_geometry()

        self.tmp_domain = (
            tmp_domain
            if tmp_domain is not None
            else solution_geometry.allocate()
        )

        super(TikhonovOperator, self).__init__(
            domain_geometry=hybrid_domain,
            range_geometry=hybrid_range,
        )

    @property
    def weights(self):
        """
        Return the mutable diagonal IRLS weights.

        The weights live in Range(L), not necessarily in Domain(A).
        """
        return self.reg_operator.weights

    def direct(self, x, out=None):
        r"""
        Apply

        .. math::

            K x = A (W L)^{-1} x.
        """
        if out is None:
            out = self.range_geometry().allocate()

        self.reg_operator.inverse(x, out=self.tmp_domain)
        self.operator.direct(self.tmp_domain, out=out)

        return out

    def adjoint(self, x, out=None):
        r"""
        Apply

        .. math::

            K^* x = (W L)^{-*} A^* x.
        """
        if out is None:
            out = self.domain_geometry().allocate()

        self.operator.adjoint(x, out=self.tmp_domain)
        self.reg_operator.inverse_adjoint(self.tmp_domain, out=out)

        return out


class WeightedStructOperator(LinearOperator):
    r"""
    Weighted structural operator for IRLS.

    This composes a structural regularisation operator ``L`` with a diagonal
    weight operator ``W``:

    .. math::

        L_{\text{weighted}} = W L.

    During IRLS, the weights approximate an L1 penalty:

    .. math::

        w_k = (|L u_{k-1}|^2 + \tau_k^2)^{-1/4}.

    For a block-valued structural operator, such as a gradient operator, the
    weights live in ``Range(L)`` and are therefore block-valued. This is handled
    by the block-aware ``DiagonalOperator``.

    Geometries
    ----------
    Domain:
        Domain(L), the physical solution space.

    Range:
        Range(L), the weighted structure space.
    """

    def __init__(
        self,
        domain_geometry,
        struct_operator=None,
        tmp_range_struct=None,
    ):
        if struct_operator is None:
            struct_operator = IdentityOperator(domain_geometry)

        self.struct_operator = struct_operator
        range_geometry = self.struct_operator.range_geometry()

        self.weight_operator: LinearOperator = IdentityOperator(
            domain_geometry=range_geometry
        )

        self.tmp_range_struct = (
            tmp_range_struct
            if tmp_range_struct is not None
            else range_geometry.allocate()
        )

        super(WeightedStructOperator, self).__init__(
            domain_geometry=domain_geometry,
            range_geometry=range_geometry,
        )

    @property
    def weights(self):
        """
        Return the mutable diagonal weights container.

        If the operator is currently unweighted, initialise explicit unit weights
        in Range(L). For gradient-like operators this may be a BlockDataContainer.
        """
        if not isinstance(self.weight_operator, DiagonalOperator):
            weights = self.range_geometry().allocate(1)
            self.weight_operator = DiagonalOperator(
                weights,
                domain_geometry=self.range_geometry(),
            )

        return self.weight_operator.diagonal

    def direct(self, x, out=None):
        r"""
        Return

        .. math::

            W L x.
        """
        if out is None:
            temp = self.struct_operator.direct(x)
            return self.weight_operator.direct(temp)

        self.struct_operator.direct(x, out=out)
        self.weight_operator.direct(out, out=out)

        return out

    def adjoint(self, x, out=None):
        r"""
        Return

        .. math::

            L^* W^* x.
        """
        self.weight_operator.adjoint(x, out=self.tmp_range_struct)

        if out is None:
            return self.struct_operator.adjoint(self.tmp_range_struct)

        self.struct_operator.adjoint(self.tmp_range_struct, out=out)

        return out
    
    def inverse(self, x, out=None):
        r"""
        Return y such that W L y = x.
        
        If the structural operator is orthogonal (L^{-1} = L^*), we compute 
        the exact inverse sequentially. Since W is diagonal, W^{-1}x is simply 
        element-wise division by the weights.
        """
        # Fast path: if L is orthogonal, L^{-1} = L^*
        if hasattr(self.struct_operator, "is_orthogonal") and self.struct_operator.is_orthogonal():
            # W^{-1} x -> divide by diagonal weights
            x.divide(self.weights, out=self.tmp_range_struct)
            
            if out is None:
                return self.struct_operator.adjoint(self.tmp_range_struct)
                
            self.struct_operator.adjoint(self.tmp_range_struct, out=out)
            return out

        else: # Use commuting approximation: (W L)^{-1} \approx L^{-1} W^{-1}
            # W^{-1} x -> divide by diagonal weights
            x.divide(self.weights, out=self.tmp_range_struct)

            if out is None:
                return self.struct_operator.inverse(self.tmp_range_struct)
            self.struct_operator.inverse(self.tmp_range_struct, out=out)
            return out
    
    def inverse_adjoint(self, x, out=None):
        r"""
        Return y such that (W L)^* y = x, which is L^* W^* y = x.
        
        If the structural operator is orthogonal (L^{-*} = L), we compute 
        the exact inverse sequentially. Since W is real and diagonal, W^{-*} 
        is simply element-wise division by the weights.
        """
        # Fast path: if L is orthogonal, L^{-*} = L
        if hasattr(self.struct_operator, "is_orthogonal") and self.struct_operator.is_orthogonal():
            if out is None:
                out = self.struct_operator.direct(x)
            else:
                self.struct_operator.direct(x, out=out)
                
            # W^{-*} (L x) -> divide by diagonal weights
            out.divide(self.weights, out=out)
            return out

        else: # Use commuting approximation: (W L)^{-*} \approx W^{-*} L^{-*}
            if out is None:
                out = self.tmp_range_struct
                self.struct_operator.inverse_adjoint(x, out=self.tmp_range_struct)
            else:
                self.struct_operator.inverse_adjoint(x, out=self.tmp_range_struct)
                out = self.tmp_range_struct

            # W^{-*} (L^* x) -> divide by diagonal weights
            out.divide(self.weights, out=out)
            return out