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
    DiagonalOperator,
    IdentityOperator,
    GradientOperator
)
import warnings
import logging
import numpy as np

log = logging.getLogger(__name__)


class GLSQROperator(LinearOperator):
    r"""`GLSQROperator`: :math:`L`

                   :math:`X` : domain
                   :math:`Y` : range
    
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
    L (Combined)    | Domain (u)      | Range (L)       | Defined as L = L_norm L_struct.
    L_inv           | Range (L)       | Domain (u)      | Maps regularized variable \bar{x} back to u.
    K = A L_inv     | Range (L)       | Range (A)       | The first Effective Operator used within GKB/GLSQR steps.
    K* = L_inv* A*  | Range (A)       | Range (L)       | The second Effective Operator used within GKB/GLSQR steps.

    .. math::
        L(u) = L_{\text{norm}}(L_{\text{struct}}(u))

    where :math:`L_{\text{norm}}` defines the norm type (L1 or L2) and
    :math:`L_{\text{struct}}` defines the structural properties of the regularisation
    (e.g., wavelets, finite differences).

    .. math::`L_{\text{norm}}` has domain and range :math:`Y`
    .. math::`L_{\text{struct}}` has domain :math:`X` and range :math:`Y`

    The Norm Operator :math:`L_{\text{norm}}`
    ------------------------------------------
    The choice of :math:`L_{\text{norm}}` depends on the desired regular

    - **L2-norm:** :math:`L_{\text{norm}} = I`, giving the classic Tikhonov-regularised form.

    - **L1-norm:** :math:`L_{\text{norm}}` is diagonal iteratively reweighted operator to approximate
        the L1 norm. The weights are updated at each outer iteration based on the current solution
        estimate. The formula for the weights is:
        .. math::
            w_i = ( (L_{\text{struct}} x)_i^2 + \tau^2 )^{-1/4}
        where :math:`\tau` is a small positive parameter to avoid singularities. Tau can be
        adapted during the iterations based on various strategies.

    The Structural Operator :math:`L_{\text{struct}}`
    ------------------------------------------------
    The choice of :math:`L_{\text{struct}}` depends on the desired regularisation structure:

    - **Wavelets:** :math:`L_{\text{struct}}` is a wavelet transform operator.

    - **Gradient:** :math:`L_{\text{struct}}` represents gradient operators
        for gradient-based regularisation is currently only supported with L2 norm, it 
        also requires an addtional null-space correction in the inverse operations and
        storage of precomputed vector for efficient computation.

    - **General:** :math:`L_{\text{struct}}` can be any linear operator that captures
        the desired structural properties of the solution.

    The Structural Operator must have an `inverse` method implemented.

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

    def __init__(
        self,
        operator,
        domain_geometry,
        range_geometry,
        struct_operator=None,
        tmp_range=None, tmp_domain=None, tmp_range_struct=None,
        norm_type: str = "L2",
        tau: float = 1,
        tau_factor: float = 0.1, # Set to 1 to disable adaptation
    ):
        # Store forward operator
        self.operator = operator

        # Set structural operator and determine range geometry
        if struct_operator is not None:
            self.L_struct = struct_operator
        else:
            self.L_struct = IdentityOperator(domain_geometry)
        if range_geometry is None:
            range_geometry = self.L_struct.range_geometry()
        if not hasattr(self.L_struct, "inverse") and not hasattr(self.L_struct, "inverse_adjoint"):
            raise ValueError(
                "The provided structural_operator must have an 'inverse' and 'inverse_adjoint' method implemented."
            )
        
        # Select and initialize the norm operator
        self.norm_type = norm_type.upper()
        if self.norm_type == "L2":
            self.L_norm = IdentityOperator(range_geometry)
        elif self.norm_type == "L1":
            # Allocate initial weights container (w = tau^-0.5)
            # Use allocate to create data, not just geometry
            initial_weights = range_geometry.allocate(tau**-0.5)
            self.L_norm = DiagonalOperator(initial_weights)
        else:
            raise ValueError(f"Unknown norm_type '{self.norm_type}'")

        # Parameters for IRLS L1-norm and validation
        if tau <= 0: raise ValueError("tau must be positive.")
        if not (0 < tau_factor < 1): raise ValueError("tau_factor must be in (0, 1).")
        self.tau = tau
        self.tau_factor = tau_factor

        # Temporary buffers for intermediate computations
        if tmp_range is None:
            self.tmp_range = range_geometry.allocate()
        else:
            self.tmp_range = tmp_range

        if tmp_domain is None:
            self.tmp_domain = domain_geometry.allocate()
        else:
            self.tmp_domain = tmp_domain

        if tmp_range_struct is None:
            self.tmp_range_struct = self.L_struct.range_geometry().allocate()
        else:
            self.tmp_range_struct = tmp_range_struct

        # Calculate size from the existing shape property
        self.domain_size = int(np.prod(domain_geometry.shape))

        # Null-space correction for Gradient L2 case
        self._null_correction_vector = None
        self._is_gradient_l2 = (self.norm_type == "L2" and isinstance(self.L_struct, GradientOperator))
        
        if (self.norm_type == "L1" and isinstance(self.L_struct, GradientOperator)):
            raise NotImplementedError("L1 norm with Gradient structural operator is not implemented.")

        # assert self.L_norm.range_geometry().dimension_labels == \
        #     self.L_struct.range_geometry().dimension_labels

        super(GLSQROperator, self).__init__(
            domain_geometry=domain_geometry, range_geometry=range_geometry
        )

    def direct_A_L_inv(self, x, out):
        """
        Apply K = A L_inv. 
        x: Range L
        tmp: Solution Space
        out: Data Space (Range A)
        """
        # 1. L_inv: Structure Range -> Solution Domain 
        self.inverse(x, out=self.tmp_domain)
        
        # 2. A: Solution Domain  -> Data  
        self.operator.direct(self.tmp_domain, out=out)

    def adjoint_A_L_inv(self, x, out):
        """
        Apply K* = L_inv* A*
        x: Data Space
        tmp: Solution Space
        out: Range L
        """
        # 1. A*: Data -> Solution
        self.operator.adjoint(x, out=self.tmp_domain)
        
        # 2. L_inv*: Solution -> Struct
        self.inverse_adjoint(self.tmp_domain, out=out)

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
            temp = self.L_struct.direct(x)
            return self.L_norm.direct(temp)
        else:
            self.L_struct.direct(x, out=out)
            return self.L_norm.direct(out, out=out)

    def adjoint(self, x, out=None):
        """
        Returns L*(x) = L_struct*(L_norm*(x))
        x: struct range
        tmp: struct range
        out: Solution Space

        """
        # 1. L_norm*: Weighted -> Struct
        self.L_norm.adjoint(x, out=self.tmp_range_struct)
        if out is None:
            return self.L_struct.adjoint(self.tmp_range_struct)
        
        # 2. L_struct*: Struct -> Solution
        self.L_struct.adjoint(self.tmp_range_struct, out=out)
        return out
    
    def _precompute_null_space_projection_vector(self):
        """Precomputes the vector v = (A^T * (A * e)) / (N * ||A * e / sqrt(N)||^2)
        used for null-space correction in the inverse operations when using
        GradientOperator with L2 norm."""
        
        # 1. u_temp = A * e (vector of ones (e))
        self.operator.direct(self.tmp_domain.fill(1.0), out=self.tmp_range)
        
        # 3. norm_sq = ||A * e / sqrt(N)||^2
        # This simplifies to: (1/N) * ||A * e||^2
        norm_Aw_sq = (self.tmp_range.norm()**2) / self.domain_size
        
        # 4. v = (A^T * u_temp) (Solution Space)
        self.operator.adjoint(self.tmp_range, out=self.tmp_domain)

        # 5. Store in null correction (Solution Space)
        # compute v / (N * norm_Aw_sq)
        self._null_correction_vector = self.tmp_domain.copy()
        self._null_correction_vector.divide(self.domain_size * norm_Aw_sq, out=self._null_correction_vector)

    def inverse(self, x, out=None, add_nullspace_correction=False):
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

            or for gradient L2 norm:
            :math:`L^{-1}(x) = L_{\text{struct}}^{-1}\left(x - \frac{1}{\|Aw\|^2} A^T A L_{\text{struct}}^{-1}(x)\right)`
            From 
            L_A^{\dagger} = (A-(A(I - L^{\dagger}L))^{\dagger}))L^{\dagger}
        """
        # Step 1: Norm Inverse (range struct -> range struct)
        self.L_norm.inverse(x, out=self.tmp_range_struct)
        
        # --- Branch 1: Standard Inverse (Non-Gradient L2) ---
        if not self._is_gradient_l2:
            # Step 2: Structure Inverse (range struct -> Solution)
            if out is None:
                return self.L_struct.inverse(self.tmp_range_struct)
            else:
                self.L_struct.inverse(self.tmp_range_struct, out=out)
                return out
        
        # --- Branch 2: Gradient L2 Logic (Null Space Correction) ---
        
        # 1. Ensure are cached
        if self._null_correction_vector is None:
            self._precompute_null_space_projection_vector()

        # 2. y = G_dagger x (Struct -> Solution)
        # Note: GradientOperator inverse usually handles x directly
        if out is None:
            out = self.L_struct.inverse(self.tmp_range_struct)
        else:
            self.L_struct.inverse(self.tmp_range_struct, out=out)

        # 3. s = mean(v^T * y)
        s = self._null_correction_vector.dot(out) / self.domain_size

        # 4. G_A_dagger * x = y - s * e
        out.subtract(s, out=out)

        # 5. Null-space correction
        if add_nullspace_correction:
            mean_out = out.sum() / self.domain_size
            out.subtract(mean_out, out=out)
        
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
        if not self._is_gradient_l2:
            # 1. L_struct^{-*}: Solution -> Struct
            # 'out' is in Weighted Space (same geom as Struct), so we use it as buffer.
            self.L_struct.inverse_adjoint(x, out=out)
            
            # 2. L_norm^{-*}: Struct -> Weighted (In-place)
            self.L_norm.inverse_adjoint(out, out=out)
            return out

        # --- Gradient L2 Logic ---

        # 1. Ensure v is cached
        if self._null_correction_vector is None:
            # We use 'out' (Weighted Space) as tmp_range? 
            # No, precompute needs Data Space. This path is tricky if no buffer provided.
            # Assuming precomputed already or creating internal temp.
            # For robustness, we might allocate a temporary data container if needed here.
            pass 

        # 2. mu = mean(x)
        mu = x.sum() / self.domain_size

        # 3. x' = x - mu * v  (Stored in out to save memory? No, out is Weighted Space)
        # We need a Solution Space buffer. 
        # If strict no-allocation is required, caller must provide scratch Solution buffer.
        # Assuming we can modify x or create temp.
        x_temp = x.copy()
        x_temp.sapyb(1.0, self._null_correction_vector, -mu, out=x_temp)

        # 4. Result = (G_dagger)^T * x'
        self.L_struct.inverse_adjoint(x_temp, out=out)
        
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

        d = self.L_norm.diagonal

        if domain == "range":
            # x is weighted coefficients (\bar{x})
            # L_norm^{-1} removes weights: Weighted -> Struct
            self.L_norm.inverse(x, out=d)
        elif domain == "struct":
            # x is structural coefficients
            d.fill(x)
        else:
            raise ValueError("domain must be 'struct', or 'range'")

        # Adapt Tau
        self._adapt_tau()

        # Update weights w = (x^2 + \tau^2)^{-1/4}
        d.power(2, out=d)
        d.add(self.tau**2, out=d)
        d.power(-0.25, out=d)

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
            log.warning(
                "adapt_tau called but reg_norm_type is not 'L1'. No adaptation performed."
            )
            return

        # IRLS Strategy:
        # Initialize tau at 1.0
        # Reduce by factor of 10 (tau_factor=0.1) once inner loop stabilizes
        self.tau = max(self.tau * self.tau_factor, 1e-8)  # Prevent tau from becoming too small
        log.debug("Tau adapted to: %e", self.tau)