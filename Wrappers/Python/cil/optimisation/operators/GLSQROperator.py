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

from cil.optimisation.operators import (
    LinearOperator,
    IRLSRegularisationOperator
)

import logging
import numpy as np

log = logging.getLogger(__name__)



class GLSQROperator(LinearOperator):
    r"""`GLSQROperator`: :math:`K = A L^{-1}` for use with `GLSQR` algorithm.
    
    Handles the transformation of non-standard Tikhonov regularization problems into
    standard form for use with the `GLSQR` algorithm.

    Unique Geometries
    -----------------
    - Solution domain (u): domain_geom_solution
    - Data range (b): range_geom_data
    - Structure space (\bar{x}): range_geom_struct

    Operator        | Domain          | Range           | Notes
    --------------- | --------------- | --------------- | ----------------------------------------------------
    A               | Domain (u)      | Range (A)       | The forward physics model.
    L_struct        | Domain (u)      | Range (L)       | Structural part (e.g., Gradient, Wavelets).
    L_norm          | Range (L)       | Range (L)       | Square/Diagonal; operates on L_struct(u).
    L (Combined)    | Domain (u)      | Range (L)       | Defined as L = L_norm L_struct. Managed by `RegularisationOperator`.
    L_inv           | Range (L)       | Domain (u)      | Maps regularized variable \bar{x} back to u.
    K = A L_inv     | Range (L)       | Range (A)       | The first Effective Operator used within GKB/GLSQR steps.
    K* = L_inv* A*  | Range (A)       | Range (L)       | The second Effective Operator used within GKB/GLSQR steps.
    
    .. math::
        K(u) = A L^{-1}(u)
    
        
    where :math:`A` defines the forward physics model, 
    :math:`L(u) = L_{\text{norm}}(L_{\text{struct}}(u))` with :math:`L_{\text{norm}}` 
    defines the norm type (L1 or L2) and :math:`L_{\text{struct}}` defines the structural 
    properties of the regularisation (e.g., wavelets, finite differences).
    """

    def __init__(
        self,
        operator,
        domain_geometry,
        range_geometry,
        struct_operator=None,
        tmp_range=None, tmp_domain=None, tmp_range_struct=None,
        norm_type: str = "L2",
        tau: float = 1,
        tau_factor: float = 0.1,
    ):
        """
        Initialisation of the GLSQROperator.

        Parameters
        ----------
        domain_geometry: CIL Geometry
            domain of the operator
        range_geometry: CIL Geometry, optional
            range of the operator, default: same as domain
        norm_type: str, optional
            Type of norm for regularisation, options are 'L2' (default) or 'L1'
        operator: LinearOperator, optional
            Forward operator :math:`A`. Required for computing certain quantities in the inverse.
        struct_operator: LinearOperator, optional
            Structural operator :math:`L_{\text{struct}}`. If None, IdentityOperator is used.
        tau: float, optional
            Smoothing parameter for IRLS L1 regularisation. Default is 1e-3
        tau_factor: float, optional
            Factor for adapting tau. Default is 0.1.
        """
        # Store forward operator
        self.operator = operator

        # Initialize the combined regularisation operator L
        self.L = IRLSRegularisationOperator(
            domain_geometry=domain_geometry,
            struct_operator=struct_operator,
            tmp_range_struct=tmp_range_struct,
            norm_type=norm_type,
            tau=tau,
            tau_factor=tau_factor
        )

        if range_geometry is None:
            range_geometry = self.L.range_geometry()

        # Temporary buffers for intermediate computations
        if tmp_range is None:
            self.tmp_range = range_geometry.allocate()
        else:
            self.tmp_range = tmp_range

        if tmp_domain is None:
            self.tmp_domain = domain_geometry.allocate()
        else:
            self.tmp_domain = tmp_domain

        # Calculate size from the existing shape property
        self.domain_size = int(np.prod(domain_geometry.shape))

        super(GLSQROperator, self).__init__(
            domain_geometry=domain_geometry, range_geometry=range_geometry
        )

    def direct(self, x, out):
        """
        Apply K = A L_inv. 
        """
        self.L.inverse(x, out=self.tmp_domain)
        self.operator.direct(self.tmp_domain, out=out)

    def adjoint(self, x, out):
        """
        Apply K* = L_inv* A*
        """
        self.operator.adjoint(x, out=self.tmp_domain)
        self.L.inverse_adjoint(self.tmp_domain, out=out)

    @property
    def reg_operator(self):
        """Descriptive alias for the regularisation operator L."""
        return self.L