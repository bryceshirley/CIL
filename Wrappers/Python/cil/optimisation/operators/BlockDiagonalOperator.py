#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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

from cil.framework import BlockDataContainer, BlockGeometry
from cil.optimisation.operators import LinearOperator

class DiagonalOperator(LinearOperator):

    r"""DiagonalOperator

    Multiplies element-wise by a fixed container, i.e. takes the `Hadamard
    product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`_ of a
    ``diagonal`` :math:`D` and the argument :math:`x`:

    .. math:: (D \circ x)_{i,j} = D_{i,j}\, x_{i,j}

    In matrix-vector terms, flattening a :math:`M \times N` container gives a
    vector of length :math:`M N`; putting it on the main diagonal of an
    :math:`M N \times M N` matrix and multiplying by the flattened :math:`x`
    reproduces the expression above, and reshaping recovers a :math:`M \times N`
    container. The operator is self-adjoint for a real diagonal; for a complex
    one the adjoint multiplies by :math:`\overline{D}`.

    This class dispatches on the type of ``diagonal``. A plain
    :class:`~cil.framework.DataContainer` is handled by
    :class:`_DiagonalOperator`. A :class:`~cil.framework.BlockDataContainer` is
    handled by :class:`_BlockDiagonalOperator`, which applies one diagonal per
    block and so is equivalent to a :class:`~cil.optimisation.operators.BlockOperator`
    with these diagonals on its main diagonal and ``ZeroOperator`` s elsewhere,
    without paying for the off-diagonal blocks. Blocks may themselves be blocks,
    to any depth.

    Parameters
    ----------
    diagonal : DataContainer or BlockDataContainer
        The values to multiply by, with the same shape as the data to be
        operated on.
    domain_geometry : ImageGeometry or BlockGeometry, optional
        Geometry of the operator domain. If ``None``, it is taken from
        ``diagonal`` -- for a block, composed from the geometries of the
        per-block operators. Default is ``None``.

    Examples
    --------
    >>> from cil.framework import ImageGeometry, BlockDataContainer
    >>> ig = ImageGeometry(4, 3)
    >>> D = DiagonalOperator(ig.allocate(2.0))
    >>> D.direct(ig.allocate(3.0)).max()
    6.0
    >>> block = BlockDataContainer(ig.allocate(2.0), ig.allocate(5.0))
    >>> DiagonalOperator(block).direct(block)[1].max()
    25.0

    """
    def __init__(self, diagonal, domain_geometry=None):
        if isinstance(diagonal, BlockDataContainer):
            self.operator = _BlockDiagonalOperator(diagonal, domain_geometry)
        else:
            if domain_geometry is None:
                domain_geometry = diagonal.geometry
            self.operator = _DiagonalOperator(diagonal, domain_geometry)
        # Take the geometry from whichever operator was chosen rather than
        # deriving it again here. For a block of blocks `diagonal.geometry` is
        # None, and reading it here left the wrapper with no domain or range.
        super(DiagonalOperator, self).__init__(
            domain_geometry=self.operator.domain_geometry(),
            range_geometry=self.operator.range_geometry())
        self.diagonal = diagonal
    def direct(self,x,out=None):
        r"""Return :math:`D \circ x`."""
        return self.operator.direct(x,out=out)

    def adjoint(self,x, out=None):
        r"""Return :math:`\overline{D} \circ x`."""
        return self.operator.adjoint(x,out=out)

    def calculate_norm(self, **kwargs):
        r""" Returns the operator norm of DiagonalOperator which is the :math:`\infty` norm of `diagonal`

        .. math:: \|D\|_{\infty} = \max_{i}\{|D_{i}|\}
        """
        return self.operator.calculate_norm(**kwargs)


class _DiagonalOperator(LinearOperator):

    r"""Element-wise multiplication by a single :class:`~cil.framework.DataContainer`.

    The non-block half of :class:`DiagonalOperator`, which is the public entry
    point and dispatches here. See that class for the mathematics.

    Parameters
    ----------
    diagonal : DataContainer
        The values to multiply by, with the same dimensions as the data to be
        operated on.
    domain_geometry : ImageGeometry, optional
        Geometry of the operator domain. If ``None``, ``diagonal.geometry`` is
        used. Default is ``None``.

    """
    def __init__(self, diagonal, domain_geometry=None):
        if domain_geometry is None:
            domain_geometry = diagonal.geometry
        super(_DiagonalOperator, self).__init__(domain_geometry=domain_geometry,
                                    range_geometry=domain_geometry)
        self.diagonal = diagonal

    def direct(self,x,out=None):
        r"""Return :math:`D \circ x`."""
        if out is None:
            return self.diagonal * x
        else:
            self.diagonal.multiply(x,out=out)
        return out

    def adjoint(self,x, out=None):
        r"""Return :math:`\overline{D} \circ x`."""
        return self.diagonal.conjugate().multiply(x,out=out)

    def calculate_norm(self, **kwargs):
        r""" Returns the operator norm of DiagonalOperator which is the :math:`\infty` norm of `diagonal`

        .. math:: \|D\|_{\infty} = \max_{i}\{|D_{i}|\}
        """
        return self.diagonal.abs().max()
    
class _BlockDiagonalOperator(LinearOperator):

    r"""Element-wise multiplication, block by block.

    The block half of :class:`DiagonalOperator`, which is the public entry point
    and dispatches here. Given a :class:`~cil.framework.BlockDataContainer`
    :math:`D = (D_1, \ldots, D_k)` it applies one
    :class:`DiagonalOperator` per block,

    .. math:: (D \circ x)_i = D_i \circ x_i,

    which is what a :class:`~cil.optimisation.operators.BlockOperator` carrying
    these diagonals on its main diagonal and ``ZeroOperator`` s off it would
    compute, at the cost of the diagonal alone. Because each block is built by
    :class:`DiagonalOperator` rather than by ``_DiagonalOperator``, a block that
    is itself a block recurses, so nesting may go to any depth.

    Domain and range are the same geometry: multiplying element-wise cannot
    change shape.

    Parameters
    ----------
    diagonal : BlockDataContainer
        The values to multiply by, with the same shape as the data to be
        operated on.
    domain_geometry : BlockGeometry, optional
        Geometry of the operator domain. If ``None``, it is composed from the
        domain geometries of the per-block operators. Note that this is not the
        same as ``diagonal.geometry``, which returns ``None`` for a block of
        blocks. Default is ``None``.

    """
    def __init__(self, diagonal, domain_geometry=None):
        self.diagonal_operator_list = [ DiagonalOperator(diagonal[i]) for i in range(len(diagonal)) ]
        if domain_geometry is None:
            # Not `diagonal.geometry`: that builds a BlockGeometry out of
            # `el.geometry.copy()` for each child, and BlockGeometry has no
            # copy(), so for a block of blocks the AttributeError is swallowed
            # and it returns None. The operator was then left with no domain and
            # no range, and dot_test -- or anything else that allocates from the
            # geometry -- raised on it. Each child operator already knows its own
            # domain at whatever depth, so compose the geometry from them.
            domain_geometry = BlockGeometry(
                *[op.domain_geometry() for op in self.diagonal_operator_list])
        super(_BlockDiagonalOperator, self).__init__(domain_geometry=domain_geometry,
                                    range_geometry=domain_geometry)
        self.diagonal = diagonal

    def direct(self,x,out=None):
        r"""Return :math:`D \circ x`."""
        if out is None:
            out = x.copy()
        for i in range(len(self.diagonal)):
            self.diagonal_operator_list[i].direct(x[i], out=out[i])
        return out

    def adjoint(self,x, out=None):
        r"""Return :math:`\overline{D} \circ x`."""
        if out is None:
            out = x.copy()
        for i in range(len(self.diagonal)):
            self.diagonal_operator_list[i].adjoint(x[i], out=out[i])
        return out

    def calculate_norm(self, **kwargs):
        r"""Return the operator norm, the :math:`\infty` norm of ``diagonal``.

        Block-diagonal, so this is the largest of the per-block norms, which is
        the same thing as the largest entry over all of the blocks:

        .. math:: \|D\|_{\infty} = \max_{k} \|D_k\|_{\infty}
                                 = \max_{k,i} \{|(D_k)_i|\}
        """
        norms = [ op.calculate_norm(**kwargs) for op in self.diagonal_operator_list ]
        return max(norms)