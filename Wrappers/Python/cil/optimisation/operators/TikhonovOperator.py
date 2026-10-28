#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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

import numpy as np

from cil.optimisation.operators import (
    LinearOperator,
    IdentityOperator,
    BlockOperator,
    ScaledOperator,
    WeightedOperator,
)
from cil.optimisation.operators.WeightedOperator import _has_own_inverse

log = logging.getLogger(__name__)

VALID_FORMS = ('block', 'standard')


def _build_tikhonov_operator(operator, struct_operator=None, regalpha=0.0,
                             form='block', solution_geometry=None):
    r"""Builds the operator :math:`K` a Krylov solver iterates on for the generalised Tikhonov problem :math:`\min_u \|Au - b\|_2^2 + \alpha^2 \|Lu\|_2^2`.
    Private: LSQR and CGLS call this from ``set_up``.

    Parameters
    ----------
    operator: `LinearOperator`
        The forward model :math:`A`.
    struct_operator: `LinearOperator`, default None
        The structural operator :math:`L`. ``None`` means the identity.
    regalpha: float, default 0.0
        The regularisation parameter :math:`\alpha`. Zero returns the bare
        ``operator`` unwrapped.
    form: {'block', 'standard'}, default 'block'
        'block' builds :math:`K = [A;\ \alpha L]` (`BlockTikhonovOperator`),
        which accepts any :math:`L` and warm starts; 'standard' builds
        :math:`K = A L^{-1}` (`TikhonovOperator`), which is cheaper but needs
        an invertible :math:`L`, and LSQR cannot warm start in it.
    solution_geometry: cil geometry, default None
        The physical solution space. Defaults to ``operator.domain_geometry()``.

    Returns
    -------
    `LinearOperator` -- the class records which form was built.
    """
    if form not in VALID_FORMS:
        raise ValueError(
            "form must be one of {}, got {!r}.".format(VALID_FORMS, form))

    if regalpha < 0:
        raise ValueError(
            "regalpha must be non-negative, got {}.".format(regalpha))

    if solution_geometry is None:
        solution_geometry = operator.domain_geometry()

    if regalpha == 0:
        return operator

    if form == 'standard':
        return TikhonovOperator(
            operator,
            solution_geometry,
            struct_operator=struct_operator,
            regalpha=regalpha,
        )

    return BlockTikhonovOperator(
        operator,
        solution_geometry,
        struct_operator=struct_operator,
        regalpha=regalpha,
    )


class BlockTikhonovOperator(BlockOperator):
    r"""Stacked Tikhonov operator :math:`K = [A;\ \alpha L]`, regularising directly in the physical solution space.
    It never forms :math:`L^{-1}`, so it is the only valid choice when
    :math:`L` is not invertible.

    Parameters
    ----------
    operator: `LinearOperator`
        The forward model :math:`A`.
    solution_geometry: cil geometry
        The physical solution space, ``Domain(A)``.
    struct_operator: `LinearOperator`, default None
        The structural operator :math:`L`. ``None`` means the identity.
    regalpha: float, default 0.0
        The regularisation parameter :math:`\alpha`.
    tmp_domain: DataContainer, default None
        Accumulator the allocation-free adjoint needs when :math:`L \neq I`;
        pass one to borrow it instead of allocating.
    """

    def __init__(self, operator, solution_geometry, struct_operator=None,
                 regalpha=0.0, tmp_domain=None):
        self.operator = operator

        self.struct_is_identity = (struct_operator is None
                                   or isinstance(struct_operator,
                                                 IdentityOperator))
        if struct_operator is None:
            struct_operator = IdentityOperator(solution_geometry)
        self.struct_operator = struct_operator
        self.scaled_reg_operator = ScaledOperator(struct_operator, regalpha)

        # Accumulator for the allocation-free adjoint; needed whenever L != I.
        if tmp_domain is None and not self.struct_is_identity:
            tmp_domain = solution_geometry.allocate(0)
        self.tmp_domain = tmp_domain

        super(BlockTikhonovOperator, self).__init__(
            self.operator,
            self.scaled_reg_operator,
        )

    @property
    def regalpha(self):
        """The current Tikhonov regularisation parameter."""
        return self.scaled_reg_operator.scalar

    @regalpha.setter
    def regalpha(self, value):
        """Update the regularisation parameter in place."""
        self.scaled_reg_operator.scalar = value

    def adjoint(self, x, out=None):
        r"""Returns :math:`K^* x = A^* x_0 + \alpha L^* x_1`.
        Overrides ``BlockOperator.adjoint``, which allocates a domain
        container on every call; this writes through ``out`` and the
        ``tmp_domain`` accumulator instead. Only the two-row shape is handled
        here; anything else falls back to the base class.

        Parameters
        ----------
        x: BlockDataContainer
            Element in the range of the operator.
        out: DataContainer, default None
            If out is not None the result is filled in out, otherwise a new
            object is returned.

        Returns
        -------
        DataContainer containing the result.
        """
        if self.shape != (2, 1):
            return super(BlockTikhonovOperator, self).adjoint(x, out=out)

        if out is None:
            out = self.domain_geometry().allocate(0)

        self.operator.adjoint(x[0], out=out)

        if self.tmp_domain is None:
            # L = I, so the second row is a plain scaling and needs no
            # accumulator at all.
            return out.sapyb(1.0, x[1], self.regalpha, out=out)

        self.struct_operator.adjoint(x[1], out=self.tmp_domain)
        return out.sapyb(1.0, self.tmp_domain, self.regalpha, out=out)


class TikhonovOperator(LinearOperator):
    r"""Standard-form Tikhonov operator :math:`K = A L^{-1}`, iterating in ``Range(L)``; the physical solution is :math:`u = L^{-1} x`.
    :math:`\alpha` is not applied here -- in standard form it enters the
    solver's own recurrence -- and is stored only so callers can check the
    two agree. A plain :math:`L` must be orthogonal and square, so that
    :math:`L^{-1} = L^*`; a `WeightedOperator` is inverted through its own
    ``inverse`` instead. LSQR cannot warm start in this form: the penalty
    lands on the step, not the solution.

    An :math:`L` with a non-empty ``null_space()``, such as a ``GradientOperator``,
    is handled by deflation: with :math:`N` spanning the null space and :math:`P`
    the orthogonal projector onto the complement of :math:`\mathrm{Range}(AN)`,
    the solver iterates on :math:`K = P A L^+` against :math:`P b`, and
    :meth:`map_to_solution` adds back the null-space component that best fits the data.

    Parameters
    ----------
    operator: `LinearOperator`
        The forward model :math:`A`.
    solution_geometry: cil geometry
        The physical solution space, ``Domain(L)``.
    struct_operator: `LinearOperator`, default None
        The structural operator :math:`L`. ``None`` means the identity.
    regalpha: float, default 0.0
        The regularisation parameter :math:`\alpha`, stored for bookkeeping.
    tmp_domain: DataContainer, default None
        Staging buffer needed when :math:`L \neq I`; pass one to borrow it
        instead of allocating.
    """

    def __init__(self, operator, solution_geometry, struct_operator=None,
                 regalpha=0.0, tmp_domain=None):
        self.operator = operator
        self.regalpha = regalpha

        self.struct_is_identity = (struct_operator is None
                                   or isinstance(struct_operator,
                                                 IdentityOperator))
        if struct_operator is None:
            struct_operator = IdentityOperator(solution_geometry)
        self.struct_operator = struct_operator

        # A WeightedOperator carries a true inverse pair and guards it
        # itself; any other L is inverted as L^*, guarded in _inverse. Either
        # way the check fires where the inverse is formed, not here.
        self._weighted_operator = (struct_operator
                                   if isinstance(struct_operator,
                                                 WeightedOperator) else None)

        # With L = I the transformation is K = A exactly: no staging buffer,
        # and the degenerate path costs the same as an unregularised solve.
        if tmp_domain is None and not self.struct_is_identity:
            tmp_domain = solution_geometry.allocate(0)
        self.tmp_domain = tmp_domain

        # The null space of WL is that of L; deflate A along its image
        plain = (struct_operator.operator if self._weighted_operator is not None
                 else struct_operator)
        self._null_basis = (plain.null_space() if hasattr(plain, 'null_space')
                            else [])
        self._null_images = [operator.direct(n) for n in self._null_basis]
        self._null_gram = np.array([[float(a.dot(b)) for b in self._null_images]
                                    for a in self._null_images])
        self.tmp_range = (operator.range_geometry().allocate(0)
                          if self._null_basis else None)

        super(TikhonovOperator, self).__init__(
            domain_geometry=struct_operator.range_geometry(),
            range_geometry=operator.range_geometry(),
        )

    def _inverse(self, x, out=None):
        r"""Apply :math:`L^{-1}`: the weighted wrapper's or the operator's own
        inverse, else :math:`L^*` for a plain orthogonal, square L."""
        if self._weighted_operator is not None:
            return self._weighted_operator.inverse(x, out=out)
        if _has_own_inverse(self.struct_operator):
            return self.struct_operator.inverse(x, out=out)
        WeightedOperator._require_invertible(self.struct_operator)
        return self.struct_operator.adjoint(x, out=out)

    def _inverse_adjoint(self, x, out=None):
        r"""Apply :math:`L^{-*}`: the weighted wrapper's true inverse adjoint,
        or :math:`L` itself, since :math:`(L^{-1})^* = (L^*)^* = L` for an
        orthogonal square L."""
        if self._weighted_operator is not None:
            return self._weighted_operator.inverse_adjoint(x, out=out)
        if _has_own_inverse(self.struct_operator):
            return self.struct_operator.inverse_adjoint(x, out=out)
        WeightedOperator._require_invertible(self.struct_operator)
        return self.struct_operator.direct(x, out=out)

    def solution_geometry(self):
        """The physical solution space, Domain(L)."""
        return self.struct_operator.domain_geometry()

    def map_from_solution(self, u, out=None):
        r"""Map a physical solution into the iterate space: :math:`x = L u`."""
        return self.struct_operator.direct(u, out=out)

    def map_to_solution(self, x, out=None, data=None):
        r"""Map an iterate back to the physical solution: :math:`u = L^{-1} x`.
        When :math:`L` has a null space, the component in it that best fits
        ``data`` is added, so ``data`` is then required."""
        out = self._inverse(x, out=out)
        if not self._null_basis:
            return out
        if data is None:
            raise ValueError(
                "struct_operator has a null space: pass the data to "
                "map_to_solution to recover the component of u in it.")
        self.operator.direct(out, out=self.tmp_range)
        self.tmp_range.sapyb(-1.0, data, 1.0, out=self.tmp_range)
        for n, c in zip(self._null_basis,
                        self._null_coefficients(self.tmp_range)):
            out.sapyb(1.0, n, c, out=out)
        return out

    def _null_coefficients(self, y):
        """The coefficients of the least-squares fit of y by A N."""
        return np.linalg.solve(self._null_gram,
                               [float(a.dot(y)) for a in self._null_images])

    def _deflate(self, y, out=None):
        r"""Apply :math:`P`, removing from ``y`` its least-squares fit by
        :math:`A N`; the identity when :math:`L` has no null space."""
        if out is None:
            out = y.copy()
        elif out is not y:
            out.fill(y)
        if self._null_basis:
            for a, c in zip(self._null_images, self._null_coefficients(y)):
                out.sapyb(1.0, a, -c, out=out)
        return out

    def direct(self, x, out=None):
        r"""Apply :math:`K x = P A L^{-1} x`."""
        if out is None:
            out = self.range_geometry().allocate(0)

        if self.tmp_domain is None:
            # L = I, so K = A.
            return self.operator.direct(x, out=out)

        self._inverse(x, out=self.tmp_domain)
        self.operator.direct(self.tmp_domain, out=out)
        return self._deflate(out, out=out)

    def adjoint(self, x, out=None):
        r"""Apply :math:`K^* x = L^{-*} A^* P x`."""
        if out is None:
            out = self.domain_geometry().allocate(0)

        if self.tmp_domain is None:
            return self.operator.adjoint(x, out=out)

        if self._null_basis:
            x = self._deflate(x, out=self.tmp_range)

        self.operator.adjoint(x, out=self.tmp_domain)
        return self._inverse_adjoint(self.tmp_domain, out=out)
