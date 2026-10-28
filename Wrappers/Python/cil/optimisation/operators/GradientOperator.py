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

from cil.optimisation.operators import LinearOperator
from cil.optimisation.operators import FiniteDifferenceOperator
from cil.framework import BlockGeometry, ImageGeometry, cilacc
import logging
from cil.utilities.multiprocessing import NUM_THREADS
import numpy as np
import scipy.fft

NEUMANN = 'Neumann'
PERIODIC = 'Periodic'
C = 'c'
NUMPY = 'numpy'
CORRELATION_SPACE = "Space"
CORRELATION_SPACECHANNEL = "SpaceChannels"
log = logging.getLogger(__name__)


class GradientOperator(LinearOperator):

    r"""
    Gradient Operator: Computes first-order forward/backward differences on
    2D, 3D, 4D ImageData under Neumann/Periodic boundary conditions

    Parameters
    ----------
    domain_geometry: ImageGeometry
        Set up the domain of the function
    method: str, default 'forward'
        Accepts: 'forward', 'backward', 'centered', note C++ optimised routine only works with 'forward'
    bnd_cond: str, default,  'Neumann'
        Set the boundary conditions to use 'Neumann' or 'Periodic'. Under
        forward differences the operator has a spectral :meth:`inverse`.
    **kwargs:
        correlation: str, default 'Space'
            'Space' will compute the gradient on only the spatial dimensions, 'SpaceChannels' will include the channel dimension direction
        backend: str, default 'c'
            'c' or 'numpy', defaults to 'c' if correlation is 'SpaceChannels' or channels = 1
        num_threads: int
            If backend is 'c' specify the number of threads to use. Default is number of cpus/2
        split: boolean
            If 'True', and backend 'c' will return a BlockDataContainer with grouped spatial domains. i.e. [Channel, [Z, Y, X]], otherwise [Channel, Z, Y, X]

    Returns
    -------
    BlockDataContainer
        a BlockDataContainer containing images of the derivatives order given by `dimension_labels`
        i.e. ['horizontal_y','horizontal_x'] will return [d('horizontal_y'), d('horizontal_x')]


    Example
    -------

    2D example

    .. math::
       :nowrap:

        \begin{eqnarray}
        \nabla : X \rightarrow Y\\
        u \in X, \nabla(u) &=& [\partial_{y} u, \partial_{x} u]\\
        u^{*} \in Y, \nabla^{*}(u^{*}) &=& \partial_{y} v1 + \partial_{x} v2
        \end{eqnarray}


    """

    #kept here for backwards compatbility
    CORRELATION_SPACE = CORRELATION_SPACE
    CORRELATION_SPACECHANNEL = CORRELATION_SPACECHANNEL

    def __init__(self, domain_geometry, method = 'forward', bnd_cond = 'Neumann', **kwargs):
        # Default backend = C
        backend = kwargs.get('backend',C)

        # Default correlation for the gradient coupling
        self.correlation = kwargs.get('correlation',CORRELATION_SPACE)

        # Add assumed attributes if there is no CIL geometry (i.e. SIRF objects)
        if not hasattr(domain_geometry, 'channels'):
            domain_geometry.channels = 1

        if not hasattr(domain_geometry, 'dimension_labels'):
            domain_geometry.dimension_labels = [None]*len(domain_geometry.shape)

        self.method = method
        self.bnd_cond = bnd_cond

        if backend == C:
            if self.correlation == CORRELATION_SPACE and domain_geometry.channels > 1:
                backend = NUMPY
                log.warning("C backend cannot use correlation='Space' on multi-channel dataset - defaulting to `numpy` backend")
            elif domain_geometry.dtype != np.float32:
                backend = NUMPY
                log.warning("C backend is only for arrays of datatype float32 - defaulting to `numpy` backend")
            elif method != 'forward':
                backend = NUMPY
                log.warning("C backend is only implemented for forward differences - defaulting to `numpy` backend")
        if backend == NUMPY:
            self.operator = Gradient_numpy(domain_geometry, bnd_cond=bnd_cond, **kwargs)
        else:
            self.operator = Gradient_C(domain_geometry, bnd_cond=bnd_cond, **kwargs)

        super(GradientOperator, self).__init__(domain_geometry=domain_geometry,
                                       range_geometry=self.operator.range_geometry())

        # the differentiated axes and their voxel sizes
        self._axes = list(self.operator.ind)
        spacing = getattr(domain_geometry, 'spacing',
                          [1] * len(domain_geometry.shape))
        self._spacing = [spacing[axis] for axis in self._axes]
        self.num_threads = kwargs.get('num_threads', NUM_THREADS)
        self._spectral = None


    def direct(self, x, out=None):
        """
        Computes the first-order forward differences

        Parameters
        ----------
        x : ImageData
        out : BlockDataContainer, optional
            pre-allocated output memory to store result

        Returns
        -------
        BlockDataContainer
            result data if `out` not specified
        """
        return self.operator.direct(x, out=out)


    def adjoint(self, x, out=None):
        """
        Computes the first-order backward differences

        Parameters
        ----------
        x : BlockDataContainer
            Gradient images for each dimension in ImageGeometry domain
        out : ImageData, optional
            pre-allocated output memory to store result

        Returns
        -------
        ImageData
            result data if `out` not specified
        """

        return self.operator.adjoint(x, out=out)

    def has_inverse(self):
        """True for forward differences under 'Neumann' or 'Periodic'
        boundary conditions, which carry a spectral :meth:`inverse`."""
        return (self.method == 'forward'
                and self.bnd_cond in (NEUMANN, PERIODIC)
                and not getattr(self.operator, 'split', False))

    def inverse(self, x, out=None):
        r"""
        The Moore-Penrose pseudoinverse :math:`\nabla^+ =
        (\nabla^T\nabla)^+\nabla^T`. The null space of the gradient is the
        constants (per channel under ``correlation='Space'``, see
        :meth:`null_space`), so :math:`\nabla^+\nabla u` is :math:`u` with
        its mean removed.

        Parameters
        ----------
        x : BlockDataContainer
            Element in the range of the operator.
        out : ImageData, optional
            pre-allocated output memory to store result

        Returns
        -------
        ImageData
            result data if `out` not specified
        """
        out = self.adjoint(x, out=out)
        out.fill(self._get_spectral().solve(out.as_array()))
        return out

    def inverse_adjoint(self, x, out=None):
        r"""
        The adjoint of :meth:`inverse`, :math:`\nabla(\nabla^T\nabla)^+`.

        Parameters
        ----------
        x : ImageData
            Element in the domain of the operator.
        out : BlockDataContainer, optional
            pre-allocated output memory to store result

        Returns
        -------
        BlockDataContainer
            result data if `out` not specified
        """
        smoothed = x.copy()
        smoothed.fill(self._get_spectral().solve(x.as_array()))
        return self.direct(smoothed, out=out)

    def null_space(self):
        """
        A basis of the null space of the operator, as a list of
        ImageData: the constants, one per channel when the channel axis is
        not differentiated; empty when the gradient has no :meth:`inverse`.
        """
        if not self.has_inverse():
            return []
        geometry = self.domain_geometry()
        shape = geometry.shape
        free_axes = [axis for axis, size in enumerate(shape)
                     if size > 1 and axis not in self._axes]
        basis = []
        for index in np.ndindex(*[shape[axis] for axis in free_axes]):
            arr = np.zeros(shape, dtype=geometry.dtype)
            sl = [slice(None)] * len(shape)
            for axis, i in zip(free_axes, index):
                sl[axis] = i
            arr[tuple(sl)] = 1
            element = geometry.allocate(None)
            element.fill(arr)
            basis.append(element)
        return basis

    def _get_spectral(self):
        if not self.has_inverse():
            raise NotImplementedError(
                "inverse() needs forward differences with bnd_cond "
                "'Neumann' or 'Periodic', got method={!r}, bnd_cond={!r}."
                .format(self.method, self.bnd_cond))
        if self._spectral is None:
            self._spectral = _SpectralLaplacian(
                self.domain_geometry().shape, self._axes, self._spacing,
                self.bnd_cond, np.dtype(self.domain_geometry().dtype),
                self.num_threads)
        return self._spectral


    def calculate_norm(self):

        r"""
        Returns the analytical norm of the GradientOperator.

        .. math::

            (\partial_{z}, \partial_{y}, \partial_{x}) &= \sqrt{\|\partial_{z}\|^{2} + \|\partial_{y}\|^{2} + \|\partial_{x}\|^{2} } \\
            &=  \sqrt{ \frac{4}{h_{z}^{2}} + \frac{4}{h_{y}^{2}} + \frac{4}{h_{x}^{2}}}


        Where the voxel sizes in each dimension are equal to 1 this simplifies to:

          - 2D geometries :math:`norm = \sqrt{8}`
          - 3D geometries :math:`norm = \sqrt{12}`

        """

        if self.correlation==CORRELATION_SPACE and self._domain_geometry.channels > 1:
            norm = np.array(self.operator.voxel_size_order[1::])
        else:
            norm = np.array(self.operator.voxel_size_order)

        norm = 4 / (norm * norm)

        return np.sqrt(norm.sum())


class _SpectralLaplacian:
    r"""
    Applies :math:`(\nabla^T\nabla)^+` for forward differences, with
    eigenvalues :math:`\sum 4\sin^2(\pi k/(m n))/h^2` over the
    differentiated axes: Neumann diagonalises in the DCT-II basis
    (:math:`m = 2`), periodic in the Fourier basis (:math:`m = 1`), on real
    data through ``rfftn``. The zero eigenvalue, the constants, is dropped.
    """

    def __init__(self, shape, axes, spacing, bnd_cond, dtype, workers):
        self.shape = tuple(shape)
        self.axes = tuple(axes)
        self.bnd_cond = bnd_cond
        self.workers = workers
        ndim = len(shape)
        period = 2.0 if bnd_cond == NEUMANN else 1.0
        eigenvalues = 0
        for i, (axis, h) in enumerate(zip(self.axes, spacing)):
            n = shape[axis]
            # rfftn keeps half the spectrum of its last axis
            count = (n // 2 + 1 if bnd_cond == PERIODIC and i == len(axes) - 1
                     else n)
            k = np.arange(count).reshape([count if d == axis else 1
                                          for d in range(ndim)])
            eigenvalues = eigenvalues + (
                4 * np.sin(np.pi * k / (period * n)) ** 2 / h ** 2)
        with np.errstate(divide='ignore'):
            reciprocal = np.where(eigenvalues > 0, 1.0 / eigenvalues, 0.0)
        self.reciprocal = reciprocal.astype(dtype)

    def solve(self, arr):
        """Apply the pseudoinverse Laplacian to the array ``arr``."""
        if self.bnd_cond == NEUMANN:
            spectrum = scipy.fft.dctn(arr, type=2, axes=self.axes,
                                      norm='ortho', workers=self.workers)
            spectrum *= self.reciprocal
            return scipy.fft.idctn(spectrum, type=2, axes=self.axes,
                                   norm='ortho', workers=self.workers)
        spectrum = scipy.fft.rfftn(arr, axes=self.axes, workers=self.workers)
        spectrum *= self.reciprocal
        return scipy.fft.irfftn(spectrum, s=[self.shape[a] for a in self.axes],
                                axes=self.axes, workers=self.workers)


class Gradient_numpy(LinearOperator):

    def __init__(self, domain_geometry, method = 'forward', bnd_cond = 'Neumann', **kwargs):
        '''creator

        :param gm_domain: domain of the operator
        :type gm_domain: :code:`AcquisitionGeometry` or :code:`ImageGeometry`
        :param bnd_cond: boundary condition, either :code:`Neumann` or :code:`Periodic`.
        :type bnd_cond: str, optional, default :code:`Neumann`
        :param correlation: optional, :code:`SpaceChannel` or :code:`Space`
        :type correlation: str, optional, default :code:`Space`
        '''

        # Consider pseudo 2D geometries with one slice, e.g., (1,voxel_num_y,voxel_num_x)
        domain_shape = []
        self.ind = []
        for i, size in enumerate(list(domain_geometry.shape)):
            if size > 1:
                domain_shape.append(size)
                self.ind.append(i)

        # Dimension of domain geometry
        self.ndim = len(domain_shape)

        # Default correlation for the gradient coupling
        self.correlation = kwargs.get('correlation',CORRELATION_SPACE)
        self.bnd_cond = bnd_cond

        # Call FiniteDifference operator
        self.method = method
        self.FD = FiniteDifferenceOperator(domain_geometry, direction = 0, method = self.method, bnd_cond = self.bnd_cond)

        if self.correlation==CORRELATION_SPACE and 'channel' in domain_geometry.dimension_labels:
            self.ndim -= 1
            self.ind.remove(domain_geometry.dimension_labels.index('channel'))

        range_geometry = BlockGeometry(*[domain_geometry for _ in range(self.ndim) ] )

        #get voxel spacing, if not use 1s
        try:
            self.voxel_size_order = list(domain_geometry.spacing)
        except:
            self.voxel_size_order = [1]*len(domain_geometry.shape)
        super(Gradient_numpy, self).__init__(domain_geometry = domain_geometry,
                                             range_geometry = range_geometry)

        log.info("Initialised GradientOperator with numpy backend")

    def direct(self, x, out=None):
         if out is not None:
             for i, axis_index in enumerate(self.ind):
                 self.FD.direction = axis_index
                 self.FD.voxel_size = self.voxel_size_order[axis_index]
                 self.FD.direct(x, out = out[i])
             return out
         else:
             tmp = self.range_geometry().allocate()
             for i, axis_index in enumerate(self.ind):
                 self.FD.direction = axis_index
                 self.FD.voxel_size = self.voxel_size_order[axis_index]
                 tmp.get_item(i).fill(self.FD.direct(x))
             return tmp

    def adjoint(self, x, out=None):

        if out is not None:
            tmp = self.domain_geometry().allocate()
            for i, axis_index in enumerate(self.ind):
                self.FD.direction = axis_index
                self.FD.voxel_size = self.voxel_size_order[axis_index]
                self.FD.adjoint(x.get_item(i), out = tmp)
                if i == 0:
                    out.fill(tmp)
                else:
                    out += tmp
            return out
        else:
            tmp = self.domain_geometry().allocate()
            for i, axis_index in enumerate(self.ind):
                self.FD.direction = axis_index
                self.FD.voxel_size = self.voxel_size_order[axis_index]
                tmp += self.FD.adjoint(x.get_item(i))
            return tmp

import ctypes

c_float_p = ctypes.POINTER(ctypes.c_float)

cilacc.openMPtest.restypes = ctypes.c_int32
cilacc.openMPtest.argtypes = [ctypes.c_int32]

cilacc.fdiff4D.restype = ctypes.c_int32
cilacc.fdiff4D.argtypes = [ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.c_size_t,
                       ctypes.c_size_t,
                       ctypes.c_size_t,
                       ctypes.c_size_t,
                       ctypes.c_int32,
                       ctypes.c_int32,
                       ctypes.c_int32]

cilacc.fdiff3D.restype = ctypes.c_int32
cilacc.fdiff3D.argtypes = [ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.c_size_t,
                       ctypes.c_size_t,
                       ctypes.c_size_t,
                       ctypes.c_int32,
                       ctypes.c_int32,
                       ctypes.c_int32]

cilacc.fdiff2D.restype = ctypes.c_int32
cilacc.fdiff2D.argtypes = [ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float),
                       ctypes.c_size_t,
                       ctypes.c_size_t,
                       ctypes.c_int32,
                       ctypes.c_int32,
                       ctypes.c_int32]


class Gradient_C(LinearOperator):

    '''Finite Difference Operator:

            Computes first-order forward/backward differences
                     on 2D, 3D, 4D ImageData
                     under Neumann/Periodic boundary conditions'''

    def __init__(self, domain_geometry,  bnd_cond = NEUMANN, **kwargs):

        # Number of threads
        self.num_threads = kwargs.get('num_threads',NUM_THREADS)

        # Split gradients, e.g., space and channels
        self.split = kwargs.get('split',False)

        # Consider pseudo 2D geometries with one slice, e.g., (1,voxel_num_y,voxel_num_x)
        self.domain_shape = []
        self.ind = []
        self.voxel_size_order = []
        for i, size in enumerate(list(domain_geometry.shape) ):
            if size!=1:
                self.domain_shape.append(size)
                self.ind.append(i)
                self.voxel_size_order.append(domain_geometry.spacing[i])

        # Dimension of domain geometry
        self.ndim = len(self.domain_shape)

        #default is 'Neumann'
        self.bnd_cond = 0

        if bnd_cond == PERIODIC:
            self.bnd_cond = 1

        # Define range geometry
        if self.split is True and 'channel' in domain_geometry.dimension_labels:
            range_geometry = BlockGeometry(domain_geometry, BlockGeometry(*[domain_geometry for _ in range(self.ndim-1)]))
        else:
            range_geometry = BlockGeometry(*[domain_geometry for _ in range(self.ndim)])
            self.split = False

        if self.ndim == 4:
            self.fd = cilacc.fdiff4D
        elif self.ndim == 3:
            self.fd = cilacc.fdiff3D
        elif self.ndim == 2:
            self.fd = cilacc.fdiff2D
        else:
            raise ValueError('Number of dimensions not supported, expected 2, 3 or 4, got {}'.format(len(domain_geometry.shape)))

        super(Gradient_C, self).__init__(domain_geometry=domain_geometry,
                                         range_geometry=range_geometry)
        log.info("Initialised GradientOperator with C backend running with %d threads", cilacc.openMPtest(self.num_threads))

    @staticmethod
    def datacontainer_as_c_pointer(x):
        ndx = x.as_array()
        return ndx, ndx.ctypes.data_as(c_float_p)

    @staticmethod
    def ndarray_as_c_pointer(ndx):
        return ndx.ctypes.data_as(c_float_p)

    def direct(self, x, out=None):

        ndx = np.asarray(x.as_array(), dtype=np.float32, order='C')
        x_p = Gradient_C.ndarray_as_c_pointer(ndx)

        if out is None:
            out = self.range_geometry().allocate(None)

        if self.split is False:
            ndout = [el.as_array() for el in out.containers]
        else:
            ind = self.domain_geometry().dimension_labels.index('channel')
            ndout = [el.as_array() for el in out.get_item(1).containers]
            ndout.insert(ind, out.get_item(0).as_array()) #insert channels dc at correct point for channel data

        #pass list of all arguments
        arg1 = [Gradient_C.ndarray_as_c_pointer(ndout[i]) for i in range(len(ndout))]
        arg2 = [el for el in self.domain_shape]
        args = arg1 + arg2 + [self.bnd_cond, 1, self.num_threads]
        status = self.fd(x_p, *args)

        if status != 0:
            raise RuntimeError('Call to C gradient operator failed')

        for i, el in enumerate(self.voxel_size_order):
            if el != 1:
                ndout[i]/=el

        #fill back out in corerct (non-trivial) order
        if self.split is False:
            for i in range(self.ndim):
                out.get_item(i).fill(ndout[i])
        else:
            ind = self.domain_geometry().dimension_labels.index('channel')
            out.get_item(0).fill(ndout[ind])

            j = 0
            for i in range(self.ndim):
                if i != ind:
                    out.get_item(1).get_item(j).fill(ndout[i])
                    j +=1

        return out

    def adjoint(self, x, out=None):
        if out is None:
            out = self.domain_geometry().allocate(None)

        ndout = np.asarray(out.as_array(), dtype=np.float32, order='C')
        out_p = Gradient_C.ndarray_as_c_pointer(ndout)

        if self.split is False:
            ndx = [el.as_array() for el in x.containers]
        else:
            ind = self.domain_geometry().dimension_labels.index('channel')
            ndx = [el.as_array() for el in x.get_item(1).containers]
            ndx.insert(ind, x.get_item(0).as_array())

        for i, el in enumerate(self.voxel_size_order):
            if el != 1:
                ndx[i]/=el

        arg1 = [Gradient_C.ndarray_as_c_pointer(ndx[i]) for i in range(self.ndim)]
        arg2 = [el for el in self.domain_shape]
        args = arg1 + arg2 + [self.bnd_cond, 0, self.num_threads]

        status = self.fd(out_p, *args)
        if status != 0:
            raise RuntimeError('Call to C gradient operator failed')

        out.fill(ndout)

        #reset input data
        for i, el in enumerate(self.voxel_size_order):
            if el != 1:
                ndx[i]*= el

        return out
