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

import functools
from cil.optimisation.operators import LinearOperator
from cil.optimisation.operators import FiniteDifferenceOperator
from cil.optimisation.operators.DiagonalOperator import DiagonalOperator
from cil.framework import BlockGeometry, ImageGeometry, cilacc
import logging
from cil.utilities.multiprocessing import NUM_THREADS
import numpy as np

from scipy.fftpack import dctn, idctn, fftn, ifftn

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
        Set the boundary conditions to use 'Neumann' or 'Periodic'
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
    
    def inverse(self, x, out=None):
        r"""
        Pseudo-inverse of the Gradient operator.
        y = T \Lambda^{\dagger} T* G* x

        where G* is the adjoint of the Gradient operator.
        T is the orthogonal transform to diagonalise G*G
        and \Lambda^{\dagger} is the pseudo-inverse of the diagonal matrix of
        eigenvalues of G*G.

        The transform T is
        - the DCT for Neumann boundary conditions
        - the DFT for Periodic boundary conditions
        - the DST for Dirichlet boundary conditions (not implemented)
        
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
        if getattr(self.operator, 'method', 'forward') == 'centered':
            raise NotImplementedError("Spectral inverse not yet supported for centered differences.")
    
        if out is None:
            out = self.domain_geometry().allocate()
        
        # 1. Map to Image Space
        self.adjoint(x, out=out)
        
        # 2. Spectral Filter
        self._spectral_inverse_core(out, 'inverse', out=out)
        return out

    def inverse_adjoint(self, x, out=None):
        r"""
        Pseudo-inverse of the Adjoint Gradient operator.
        y = G T \Lambda^{\dagger} T* x

        where G is the Gradient operator.
        T is the orthogonal transform to diagonalise G*G
        and \Lambda^{\dagger} is the pseudo-inverse-adjoint of the diagonal matrix of
        eigenvalues of G*G.

        The transform T is
        - the DCT for Neumann boundary conditions
        - the DFT for Periodic boundary conditions
        - the DST for Dirichlet boundary conditions (not implemented)
        
        Parameters
        ----------
        x : ImageData
            Image in ImageGeometry domain
        out : BlockDataContainer, optional
            pre-allocated output memory to store result

        Returns
        -------
        BlockDataContainer
            result data if `out` not specified
        """
        if getattr(self.operator, 'method', 'forward') == 'centered':
            raise NotImplementedError("Spectral inverse not yet supported for centered differences.")

        if out is None:
            out = self.range_geometry().allocate()
            
        # 1. Filter in Image Space (Temporary ImageData)
        filtered_image = self._spectral_inverse_core(x, 'inverse_adjoint')
        
        # 2. Map back to Gradient Space
        self.direct(filtered_image, out=out)
        return out

    def _spectral_inverse_core(self, x_image_space, eig_method_name, out=None):
        """ Shared logic for spectral filtering in image domain. """
        T, T_inv, eig_op = self._spectral_data
        
        # Transform to Frequency Domain
        freq_arr = T(x_image_space.as_array())
        
        target_complex_dtype = np.complex128 if x_image_space.dtype == np.float64 else np.complex64
        
        # Allocate temporary frequency domain container with appropriate complex dtype
        tmp_freq = self.domain_geometry().allocate(dtype=target_complex_dtype)
        tmp_freq.fill(freq_arr)

        getattr(eig_op, eig_method_name)(tmp_freq, out=tmp_freq)

        # Transform back to Spatial Image Space
        spatial_arr = T_inv(tmp_freq.as_array()).real
        
        target_dtype = self.domain_geometry().dtype

        if out is None:
            out = self.domain_geometry().allocate(dtype=target_dtype)

        out.fill(spatial_arr)

        return out

    @functools.cached_property
    def _spectral_data(self):
        """Lazy cache for transforms and eigenvalues."""
        T, T_inv = self._get_transform_operators()
        eig_op = self._get_nonzero_eigenvalues()
        
        # Handle Pseudo-inverse: 0 -> inf so 1/inf = 0
        eigs = eig_op.diagonal.as_array()
        eigs[eigs == 0] = np.inf 
        
        return T, T_inv, eig_op
    
    def _get_transform_operators(self):
        """ Get the orthogonal transform operators T and T_adjoint. """
        bnd_cond = self.operator.bnd_cond
        if isinstance(bnd_cond, str):
            bnd_cond = bnd_cond.lower()

        if bnd_cond in [0, 'neumann']:
            norm = 'ortho' 
            T = lambda x: dctn(x, norm=norm)
            T_inv = lambda x: idctn(x, norm=norm)
        elif bnd_cond in [1, 'periodic']:
            T = lambda x: fftn(x)
            T_inv = lambda x: ifftn(x)
        else:
            raise NotImplementedError("Boundary condition not supported.")

        return T, T_inv
    
    def _get_nonzero_eigenvalues(self):
        """ Compute the eigenvalues of G*G and return as a DiagonalOperator.
        Multidimensional eigenvalues are summed together using vectorized operations.
        """
        bnd_cond = str(self.operator.bnd_cond).lower()
        geom = self.domain_geometry()
        shape = geom.shape
        spacing = geom.spacing
        labels = geom.dimension_labels
        
        factor = 2.0 if bnd_cond in ['0', 'neumann'] else 1.0
        grids = np.ogrid[tuple(slice(0, N) for N in shape)]
        
        # Determine active dimensions for gradient based on correlation
        active_mask = [
            not (self.correlation == "Space" and label == 'channel') and N > 1
            for N, label in zip(shape, labels)
        ]

        # Sum eigenvalues across active dimensions
        eigenvalues_array = sum(
            (4.0 * (np.sin(np.pi * g / (factor * N))**2) / (h**2))
            for g, N, h, active in zip(grids, shape, spacing, active_mask)
            if active
        )

        if isinstance(eigenvalues_array, (int, float)):
            eigenvalues_array = np.zeros(shape, dtype=getattr(geom, 'dtype', np.float32))

        eig_container = geom.allocate()
        eig_container.fill(eigenvalues_array)
        return DiagonalOperator(eig_container)

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
