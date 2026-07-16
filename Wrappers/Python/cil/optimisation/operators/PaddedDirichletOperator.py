import numpy as np
from cil.optimisation.operators import LinearOperator
from cil.framework import ImageGeometry
from cil.optimisation.operators import GradientOperator

class PaddedDirichletGradientOperator(LinearOperator):
    """
    Gradient Operator that enforces Dirichlet boundary conditions 
    by implicitly edge-padding the domain to avoid null-space boundary loss.
    """
    def __init__(self, domain_geometry, pad_width=1, **kwargs):
        self.pad_width = pad_width
        self.orig_geom = domain_geometry
        
        shape = domain_geometry.shape
        spacing = domain_geometry.spacing
        padded_shape = tuple(s + 2 * self.pad_width for s in shape)
        
        # 1. Create a padded geometry maintaining dimensions AND original spacing
        if len(shape) == 3:
            self.padded_geom = ImageGeometry(
                voxel_num_z=padded_shape[0], 
                voxel_num_y=padded_shape[1], 
                voxel_num_x=padded_shape[2],
                voxel_size_z=spacing[0],
                voxel_size_y=spacing[1],
                voxel_size_x=spacing[2]
            )
        elif len(shape) == 2:
            self.padded_geom = ImageGeometry(
                voxel_num_y=padded_shape[0], 
                voxel_num_x=padded_shape[1],
                voxel_size_y=spacing[0],
                voxel_size_x=spacing[1]
            )
        else:
            raise ValueError("Only 2D and 3D geometries are supported.")
        
        # 2. Instantiate the wrapped Dirichlet gradient
        kwargs['method'] = kwargs.get('method', 'forward')
        kwargs['bnd_cond'] = 'Dirichlet'
        self.grad_op = GradientOperator(self.padded_geom, **kwargs)
        
        super().__init__(
            domain_geometry=self.orig_geom,
            range_geometry=self.grad_op.range_geometry()
        )
        
    def direct(self, x, out=None):
        x_padded_arr = np.pad(x.as_array(), pad_width=self.pad_width, mode='edge')
        x_padded_img = self.padded_geom.allocate()
        x_padded_img.fill(x_padded_arr)
        return self.grad_op.direct(x_padded_img, out=out)
        
    def adjoint(self, y, out=None):
        grad_adj_arr = self.grad_op.adjoint(y).as_array()
        pw = self.pad_width
        inner_slices = tuple(slice(pw, -pw) for _ in range(grad_adj_arr.ndim))
        unpadded = grad_adj_arr[inner_slices].copy()
        
        for axis in range(grad_adj_arr.ndim):
            pad_slice = [slice(None)] * grad_adj_arr.ndim
            edge_slice = [slice(None)] * grad_adj_arr.ndim
            
            pad_slice[axis] = slice(0, pw)
            edge_slice[axis] = 0 
            unpadded[tuple(edge_slice)] += np.sum(grad_adj_arr[tuple(pad_slice)], axis=axis)
            
            pad_slice[axis] = slice(-pw, None)
            edge_slice[axis] = -1 
            unpadded[tuple(edge_slice)] += np.sum(grad_adj_arr[tuple(pad_slice)], axis=axis)
            
        if out is None: out = self.orig_geom.allocate()
        out.fill(unpadded)
        return out
        
    def inverse(self, y, out=None):
        inv_padded = self.grad_op.inverse(y)
        pw = self.pad_width
        inner_slices = tuple(slice(pw, -pw) for _ in range(inv_padded.ndim))
        if out is None: out = self.orig_geom.allocate()
        out.fill(inv_padded.as_array()[inner_slices])
        return out

    def inverse_adjoint(self, x, out=None):
        x_padded_arr = np.pad(x.as_array(), pad_width=self.pad_width, mode='edge')
        x_padded_img = self.padded_geom.allocate()
        x_padded_img.fill(x_padded_arr)
        return self.grad_op.inverse_adjoint(x_padded_img, out=out)
        
    def calculate_norm(self):
        return self.grad_op.calculate_norm()
        