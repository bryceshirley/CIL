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


class TikhonovOperator(LinearOperator):
    r"""
    Standard-form identity Tikhonov transformation operator.

    This operator represents

    .. math::

        K = A W^{-1},

    where ``W`` is a diagonal weighting operator on the solution space.

    For standard L2 Tikhonov regularisation, ``W`` is the identity.
    For standard L1 IRLS regularisation, ``W`` is updated externally by IRLS.

    This corresponds to the identity-structure case, i.e. ``L = I``.

    Geometries
    ----------
    Domain:
        Solution space.

    Range:
        Data space, i.e. ``Range(A)``.
    """

    def __init__(self, operator, solution_geometry, tmp_domain=None):
        self.operator = operator
        self.solution_geometry = solution_geometry

        self.weight_operator = IdentityOperator(
            domain_geometry=solution_geometry
        )

        self.tmp_domain = (
            tmp_domain
            if tmp_domain is not None
            else solution_geometry.allocate()
        )

        super(TikhonovOperator, self).__init__(
            domain_geometry=solution_geometry,
            range_geometry=operator.range_geometry(),
        )

    @property
    def weights(self):
        """
        Return the mutable diagonal weights container.

        If the operator is currently unweighted, initialise explicit unit weights.
        This allows IRLS to update the weights in-place without allocating a new
        container each iteration.
        """
        if not isinstance(self.weight_operator, DiagonalOperator):
            weights = self.domain_geometry().allocate(1)
            self.weight_operator = DiagonalOperator(
                weights,
                domain_geometry=self.domain_geometry(),
            )

        return self.weight_operator.diagonal

    def direct(self, x, out=None):
        r"""
        Apply

        .. math::

            K x = A W^{-1} x.
        """
        if out is None:
            out = self.range_geometry().allocate()

        self.weight_operator.inverse(x, out=self.tmp_domain)
        self.operator.direct(self.tmp_domain, out=out)

        return out

    def adjoint(self, x, out=None):
        r"""
        Apply

        .. math::

            K^* x = W^{-*} A^* x.
        """
        if out is None:
            out = self.domain_geometry().allocate()

        self.operator.adjoint(x, out=self.tmp_domain)
        self.weight_operator.inverse_adjoint(self.tmp_domain, out=out)

        return out


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


class HybridTikhonovOperator(LinearOperator):
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

        super(HybridTikhonovOperator, self).__init__(
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
        Return

        .. math::

            L^{-1} W^{-1} x.
        """
        if not hasattr(self.struct_operator, "inverse"):
            raise ValueError(
                "The structural operator must implement an 'inverse' method."
            )
        
        if isinstance(self.struct_operator, GradientOperator):
            if self.struct_operator.operator.bnd_cond != "Dirichlet":
                raise ValueError(
                    "WeightedStructOperator requires GradientOperator with "
                    "Dirichlet boundary conditions due to null-space properties."
                )
        self.weight_operator.inverse(x, out=self.tmp_range_struct)

        if out is None:
            return self.struct_operator.inverse(self.tmp_range_struct)

        self.struct_operator.inverse(self.tmp_range_struct, out=out)

        return out

    def inverse_adjoint(self, x, out=None):
        r"""
        Return

        .. math::

            W^{-*} L^{-*} x.

        Since ``inverse`` maps ``Range(L)`` to ``Domain(L)``,
        ``inverse_adjoint`` maps ``Domain(L)`` to ``Range(L)``.
        """
        if not hasattr(self.struct_operator, "inverse_adjoint"):
            raise ValueError(
                "The structural operator must implement an 'inverse_adjoint' method."
            )
        if isinstance(self.struct_operator, GradientOperator):
            if self.struct_operator.operator.bnd_cond != "Dirichlet":
                raise ValueError(
                    "WeightedStructOperator requires GradientOperator with "
                    "Dirichlet boundary conditions due to null-space properties."
                )

        if out is None:
            out = self.range_geometry().allocate()

        self.struct_operator.inverse_adjoint(x, out=out)
        self.weight_operator.inverse_adjoint(out, out=out)

        return out