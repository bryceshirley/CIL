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


from cil.framework.data_container import DataContainer
from cil.optimisation.operators import (
    LinearOperator,
    IdentityOperator,
    GradientOperator
)
from cil.optimisation.operators.BlockDiagonalOperator import DiagonalOperator

import warnings
import logging

log = logging.getLogger(__name__)

class IRLSRegularisationOperator(LinearOperator):
    r"""
    Iteratively Reweighted Least Squares (IRLS) Regularisation Operator combines
    structural and norm-based regularisation penalty.

    This operator converts L1-norm regularisation problems into a form that can 
    be solved using least squares solvers by introducing an iteratively updated 
    weighting scheme.

    Mathematically, it represents the operator :math:`L = L_{\text{norm}} L_{\text{struct}}`.

    The Norm Operator :math:`L_{\text{norm}}`
    ------------------------------------------
    The choice of :math:`L_{\text{norm}}` depends on the desired regularisation penalty:

    - **L2-norm:** :math:`L_{\text{norm}} = I`, giving the classic Tikhonov-regularised form.

    - **L1-norm:** :math:`L_{\text{norm}}` is a diagonal iteratively reweighted operator 
      used to approximate the L1 norm. The weights are updated at each outer iteration 
      based on the current solution estimate. The formula for the weights is:
      
      .. math::
          w_i = ( (L_{\text{struct}} x)_i^2 + \tau^2 )^{-1/4}
          
      where :math:`\tau` is a small positive parameter to avoid singularities. 
      Tau can be adapted during the iterations based on various strategies.

    The Structural Operator :math:`L_{\text{struct}}`
    ------------------------------------------------
    The choice of :math:`L_{\text{struct}}` depends on the desired regularisation structure:

    - **Wavelets:** :math:`L_{\text{struct}}` is a wavelet transform operator.

    - **Gradient:** :math:`L_{\text{struct}}` represents gradient operators 
      for gradient-based regularisation (e.g., Total Variation). Note: Gradients 
      must have Dirichlet boundary conditions, as these do not have a null-space 
      and the inverse is exactly computable with FFT-based solvers.

    - **General:** :math:`L_{\text{struct}}` can be any linear operator that captures
      the desired structural properties of the solution.

    **Note:** The Structural Operator must have an `inverse` and `inverse_adjoint` 
    method implemented.
    """
    def __init__(
        self,
        domain_geometry,
        struct_operator=None,
        tmp_range_struct=None,
        norm_type: str = "L2",
        tau: float = 1,
        tau_factor: float = 0.1,
    ):
        # Set structural operator
        if struct_operator is not None:
            self.struct_operator = struct_operator
        else:
            self.struct_operator = IdentityOperator(domain_geometry)
            
        range_geometry = self.struct_operator.range_geometry()
        
        # Validate that the structural operator has the necessary methods for inversion
        if not hasattr(self.struct_operator, "inverse") and not hasattr(self.struct_operator, "inverse_adjoint"):
            raise ValueError(
                "The provided structural_operator must have an 'inverse' and 'inverse_adjoint' method implemented."
            )
        if isinstance(self.struct_operator, GradientOperator):
            if self.struct_operator.operator.bnd_cond != 'Dirichlet':
                raise ValueError("GLSQROperator requires GradientOperator with Dirichlet boundary conditions due to its null-space properties.")
        
        # Select and initialize the norm operator
        self.norm_type = norm_type.upper()
        if self.norm_type == "L2":
            self.norm_operator = IdentityOperator(domain_geometry=range_geometry)
        elif self.norm_type == "L1":
            initial_weights = range_geometry.allocate(tau**-0.5)
            self.norm_operator = DiagonalOperator(initial_weights, domain_geometry=range_geometry)
        else:
            raise ValueError(f"Unknown norm_type '{self.norm_type}'")

        # Parameters for IRLS L1-norm and validation
        if tau <= 0: raise ValueError("tau must be positive.")
        if not (0 < tau_factor <= 1): raise ValueError("tau_factor must be in (0, 1].")
        self.tau = tau
        self.tau_factor = tau_factor

        # Temporary buffer for intermediate computations
        if tmp_range_struct is None:
            self.tmp_range_struct = range_geometry.allocate()
        else:
            self.tmp_range_struct = tmp_range_struct

        super(IRLSRegularisationOperator, self).__init__(
            domain_geometry=domain_geometry, range_geometry=range_geometry
        )

    def direct(self, x, out=None):
        r"""Returns the :math:`L(x) = L_{\text{norm}}(L_{\text{struct}}(x))`

        Parameters
        ----------
        x : DataContainer or BlockDataContainer
            Input data
        out : DataContainer or BlockDataContainer, optional
            If out is not None the output of the Operator will be filled in out, otherwise a new object is instantiated and returned. The default is None.

        Returns
        -------
        DataContainer or BlockDataContainer
            :math:`L(x) = L_{\text{norm}}(L_{\text{struct}}(x))`

        x: solution space
        out: struct range
        """
        if out is None:
            temp = self.struct_operator.direct(x)
            return self.norm_operator.direct(temp)
        else:
            self.struct_operator.direct(x, out=out)
            return self.norm_operator.direct(out, out=out)

    def adjoint(self, x, out=None):
        r"""Returns the inverse :math:`L*(x)=L_{\text{struct}}*(L_{\text{norm}}*(x))`
        
        Parameters
        ----------
        x : DataContainer or BlockDataContainer
            Input data, struct range
        out : DataContainer or BlockDataContainer, optional
            If out is not None the output of the Operator will be filled in out, otherwise a
        
        Returns
        -------
        DataContainer or BlockDataContainer
             :math:`L*(x)=L_{\text{struct}}*(L_{\text{norm}}*(x))`

        """
        # 1. L_norm*: Weighted -> Struct
        self.norm_operator.adjoint(x, out=self.tmp_range_struct)
        if out is None:
            return self.struct_operator.adjoint(self.tmp_range_struct)
        
        # 2. L_struct*: Struct -> Solution
        self.struct_operator.adjoint(self.tmp_range_struct, out=out)
        return out

    def inverse(self, x, out=None):
        r"""Returns the inverse :math:`L^{-1}(x)=L_{\text{struct}}^{-1}(L_{\text{norm}}^{-1}(x))`

        Parameters
        ----------
        x : DataContainer or BlockDataContainer
            Input data
        out : DataContainer or BlockDataContainer, optional
            If out is not None the output of the Operator will be filled in out, otherwise a new object is instantiated and returned. The default is None.

        Returns
        -------
        DataContainer or BlockDataContainer
            :math:`L^{-1}(x)=L_{\text{struct}}^{-1}(L_{\text{norm}}^{-1}(x))`
        """
        # Step 1: Norm Inverse (range struct -> range struct)
        self.norm_operator.inverse(x, out=self.tmp_range_struct)
        
        # Step 2: Structure Inverse (range struct -> Solution)
        if out is None:
            return self.struct_operator.inverse(self.tmp_range_struct)
        else:
            self.struct_operator.inverse(self.tmp_range_struct, out=out)
            return out
        
    def inverse_adjoint(self, x, out):
        r"""Returns the adjoint of the inverse :math:`L^{-*}(x) = L_{\text{norm}}^{-*}(L_{\text{struct}}^{-*}(x))`

        Parameters
        ----------
        x : DataContainer or BlockDataContainer
            Input data
        out : DataContainer or BlockDataContainer, optional
            If out is not None the output of the Operator will be filled in out, 
            otherwise a new object is instantiated and returned. The default is None.

        Returns
        -------
        DataContainer or BlockDataContainer
            :math:`L^{-*}(x) = L_{\text{norm}}^{-*}(L_{\text{struct}}^{-*}(x))`
        """
        # 1. struct_operator^{-*}: Solution -> Struct
        # 'out' is in Weighted Space (same geom as Struct), so we use it as buffer.
        self.struct_operator.inverse_adjoint(x, out=out)
        
        # 2. norm_operator^{-*}: Struct -> Weighted
        self.norm_operator.inverse_adjoint(out, out=out)
        return out

    def update_weights(self, x: DataContainer, domain: str = "struct"):
        """
        Update DiagonalOperator weights for IRLS L1 regularisation.

        .. math::
            w = (x^2 + \tau^2)^{-1/4}

        Where :math:`\tau` is a small positive parameter to avoid singularities.

        Parameters
        ----------
        x : DataContainer
            Current solution estimate.
        domain : {'image', 'struct', 'range'}, optional
            Defines the mathematical space of ``x`` to ensure weights are calculated
            from the structural coefficients:
            - ``'struct'``: ``x`` is in structural/transform space; use directly.
            - ``'range'``: ``x`` is in weighted range space; apply :math:`L_{norm}^{-1}`.
        adapt_tau : bool, optional
            If True, adapt the tau parameter based on the current solution. Default is False.
        """
        if self.norm_type != "L1":
            warnings.warn("update_weights called but norm_type is not 'L1'.")
            return

        d = self.norm_operator.diagonal

        if domain == "range":
            # x is weighted coefficients (\bar{x})
            # L_norm^{-1} removes weights: Weighted -> Struct
            self.norm_operator.inverse(x, out=d)
        elif domain == "struct":
            # x is structural coefficients
            d.fill(x)
        else:
            raise ValueError("domain must be 'struct', or 'range'")

        # Update weights w = (x^2 + \tau^2)^{-1/4}
        d.power(2, out=d)
        d.add(self.tau**2, out=d)
        d.power(-0.25, out=d)

        # Adapt Tau
        self._adapt_tau()

    def _adapt_tau(self):
        """Adapts the smoothing parameter tau based on various strategies.

        Strategies:
        - 'factor': Implementation of the strategy from [1] and [2].
          Tau is reduced by a factor of 10 once the objective/solution
          ceases to change significantly.
        - NOT implemented 'Daubechies': Adapts tau using the Daubechies et al. (2010) non-increasing sequence.
          Requires self.k_sparsity to be set.

        References
        ----------
        .. [1] R. Chartrand, "Exact Reconstruction of Sparse Signals via Nonconvex
           Minimization," IEEE Signal Processing Letters, vol. 14, no. 10,
           pp. 707-710, Oct. 2007. doi: 10.1109/LSP.2007.898300.

        .. [2] R. Chartrand and Wotao Yin, "Iteratively reweighted algorithms for
           compressive sensing," 2008 IEEE International Conference on Acoustics,
           Speech and Signal Processing, Las Vegas, NV, USA, 2008, pp. 3869-3872.
           doi: 10.1109/ICASSP.2008.4518498.

        .. [3] Daubechies, I., et al. "Iteratively reweighted least squares
           minimization for sparse recovery," CPAM, 2010.
        """
        if self.norm_type.upper() != "L1":
            log.warning("adapt_tau called but reg_norm_type is not 'L1'. No adaptation performed.")
            return
        
        # IRLS Strategy:
        # Initialize tau at 1.0
        # Reduce by factor of 10 (tau_factor=0.1) once inner loop stabilizes
        self.tau = max(self.tau * self.tau_factor, 1e-8) # Prevent tau from becoming too small
        log.debug("Tau adapted to: %e", self.tau)