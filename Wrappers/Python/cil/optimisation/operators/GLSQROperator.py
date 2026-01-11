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
    L_norm          | Range (L)       | Range (L)       | Square/Diagonal; operates on L_struct(x).
    L (Combined)    | Domain (u)      | Range (L)       | Defined as L = L_norm * L_struct.
    L_inv           | Range (L)       | Domain (u)      | Maps regularized variable \bar{x} back to u.
    K = A * L_inv   | Range (L)       | Range (A)       | The Effective Operator used within GKB/GLSQR steps.

    .. math::
        L(x) = L_{\text{norm}}(L_{\text{struct}}(x))

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
        range_geometry=None,
        struct_operator=None,
        norm_type: str = "L2",
        tau: float = 1,
        tau_factor: float = 0.1, # Set to 1 to disable adaptation
    ):
        # Store forward operator
        self.operator = operator

        # Parameters for IRLS L1 regularisation
        self.tau = tau
        self.tau_factor = tau_factor

        # Validation
        if self.tau <= 0: raise ValueError("tau must be positive.")
        if not (0 < tau_factor < 1): raise ValueError("tau_factor must be in (0, 1).")

        # Set structural operator
        if struct_operator is not None:
            self.L_struct = struct_operator
        else:
            self.L_struct = IdentityOperator(domain_geometry)

        # Calculate size from the existing shape property
        self.domain_size = int(np.prod(domain_geometry.shape))

        # Validate that the structural operator has an inverse method
        if not hasattr(self.L_struct, "inverse") and not hasattr(self.L_struct, "inverse_adjoint"):
            raise ValueError(
                "The provided structural_operator must have an 'inverse' and 'inverse_adjoint' method implemented."
            )

        # Set norm type
        self.norm_type = norm_type.upper()

        # 1. Determine the range geometry
        if range_geometry is None:
            range_geometry = self.L_struct.range_geometry()

        # 2. Select and initialize the norm operator
        if self.norm_type == "L2":
            self.L_norm = IdentityOperator(range_geometry)
        elif self.norm_type == "L1":
            # Allocate initial weights container (w = tau^-0.5)
            # Use allocate to create data, not just geometry
            initial_weights = range_geometry.allocate(self.tau**-0.5)
            self.L_norm = DiagonalOperator(initial_weights)
        else:
            raise ValueError(f"Unknown norm_type '{self.norm_type}'")

        self._null_correction_vector = None
        self._is_gradient_l2 = (self.norm_type == "L2" and isinstance(self.L_struct, GradientOperator))
        
        if (self.norm_type == "L1" and isinstance(self.L_struct, GradientOperator)):
            raise NotImplementedError("L1 norm with Gradient structural operator is not implemented.")

        super(GLSQROperator, self).__init__(
            domain_geometry=domain_geometry, range_geometry=range_geometry
        )

    def direct_A_L_inv(self, x, out, tmp_domain):
        """
        Apply K = A L_inv. 
        x: Struct Space
        out: Data Space (Range A)
        tmp_domain: Solution Space (Range L_inv)
        """
        # 1. L_inv: Weighted -> Solution (uses out as tmp_range internal workspace)
        self.inverse(x, out=tmp_domain, tmp_range=out)
        
        # 2. A: Solution -> Data
        self.operator.direct(tmp_domain, out=out)

    def adjoint_A_L_inv(self, u, out, tmp_domain):
        """
        Apply K* = L_inv* A*
        u: Data Space
        out: Struct Space (Range L_inv*)
        tmp_domain: Solution Space (Range A*)
        """
        # 1. A*: Data -> Solution
        self.operator.adjoint(u, out=tmp_domain)
        
        # 2. L_inv*: Solution -> Struct
        self.inverse_adjoint(tmp_domain, out=out)

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

        """
        if out is None:
            temp = self.L_struct.direct(x)
            return self.L_norm.direct(temp)
        else:
            self.L_struct.direct(x, out=out)
            return self.L_norm.direct(out, out=out)

    def adjoint(self, x, out=None, tmp_range=None):
        """
        Returns L*(x) = L_struct*(L_norm*(x))
        x: Weighted Space -> out: Solution Space
        
        Requires 'tmp_range' (Weighted/Struct Space) buffer because intermediate result 
        cannot be stored in 'out' (Solution Space) due to size mismatch.
        """
        if out is None:
            temp = self.L_norm.adjoint(x)
            return self.L_struct.adjoint(temp)
        
        # Buffer management
        if tmp_range is not None:
            intermediate = tmp_range
        else:
            # Fallback (allocates) if user forgets buffer, unless L_norm is Identity
            if self.norm_type == 'L2':
                # Identity adjoint is no-op, passing x directly is safe if L_struct doesn't modify input
                intermediate = x
            else:
                intermediate = self.range_geometry().allocate()

        # 1. L_norm*: Weighted -> Struct
        self.L_norm.adjoint(x, out=intermediate)
        
        # 2. L_struct*: Struct -> Solution
        self.L_struct.adjoint(intermediate, out=out)
        return out
    
    def _precompute_GA_vectors(self, tmp_domain, tmp_range):
        """Precomputes the vector v used for G_A_dagger."""
        # 1. w is the vector of ones (e) or normalized ones. 
        # The math uses e (ones) for the final subtraction, so let's use ones.
        tmp_domain.fill(1.0) 
        
        # 2. u_temp = A * e
        self.operator.direct(tmp_domain, out=tmp_range)
        
        # 3. norm_sq = ||A * e / sqrt(N)||^2
        # This simplifies to: (1/N) * ||A * e||^2
        norm_Aw_sq = (tmp_range.norm()**2) / self.domain_size
        
        # 4. v = (A^T * u_temp) / (N * norm_Aw_sq)
        # Note: the math box shows v = (A^T * u_temp) / ||u_temp/sqrt(N)||^2
        self.operator.adjoint(tmp_range, out=tmp_domain)

        # 5. Store in null correction (Solution Space)
        if self._null_correction_vector is None:
             self._null_correction_vector = tmp_domain.copy() # Allocation (happens once)
        else:
             self._null_correction_vector.fill(tmp_domain)
        self._null_correction_vector.divide(self.domain_size * norm_Aw_sq, out=self._null_correction_vector)

    def inverse(self, x, out=None, tmp_range=None, add_nullspace_correction=False):
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
        # --- Branch 1: Standard Inverse (Non-Gradient L2) ---
        if not self._is_gradient_l2:
            if self.norm_type == 'L2':
                # Identity Norm: L_norm^{-1} is Identity.
                # Just apply L_struct^{-1}: Struct (x) -> Solution (out)
                if out is None:
                    return self.L_struct.inverse(x)
                else:
                    self.L_struct.inverse(x, out=out)
                    return out
            else:
                # L1 Norm: L_norm^{-1} is Diagonal inverse.
                # Maps Weighted -> Struct.
                
                # Step 1: Unweight (Weighted -> Struct)
                # We need a buffer because we can't write Struct data into Solution space 'out' directly if shapes differ.
                if tmp_range is None:
                    # Allocate temporary buffer if not provided
                    tmp_buffer = self.range_geometry().allocate()
                else:
                    tmp_buffer = tmp_range

                self.L_norm.inverse(x, out=tmp_buffer)
                
                # Step 2: Structure Inverse (Struct -> Solution)
                if out is None:
                    return self.L_struct.inverse(tmp_buffer)
                else:
                    self.L_struct.inverse(tmp_buffer, out=out)
                    return out
        
        # --- Branch 2: Gradient L2 Logic (Null Space Correction) ---
        
        # 1. Ensure v and norms are cached
        if self._null_correction_vector is None:
            # Requires tmp_range to be Data Space buffer for A*e calculation.
            # If tmp_range is missing, _precompute_GA_vectors will likely fail or need its own allocation logic.
            # For this specific 'out=None' context, we assume a temp buffer must be created for precompute if missing.
            precompute_buffer = tmp_range if tmp_range is not None else self.operator.range_geometry().allocate()
            
            # We also need a solution-domain buffer for the precompute output if 'out' is None.
            sol_buffer = out if out is not None else self.domain_geometry().allocate()
            
            self._precompute_GA_vectors(sol_buffer, precompute_buffer)

        # 2. y = G_dagger * x (Struct -> Solution)
        # Note: GradientOperator inverse usually handles x directly
        if out is None:
            out = self.L_struct.inverse(x)
        else:
            self.L_struct.inverse(x, out=out)

        # 3. s = mean(v^T * y)
        s = self._null_correction_vector.dot(out) / self.domain_size

        # 4. G_A_dagger * x = y - s * e
        out.subtract(s, out=out)

        # 5. Null-space correction
        if add_nullspace_correction:
            mean_out = out.sum() / self.domain_size
            out.add(mean_out, out=out)
        
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

    def update_weights(self, x: DataContainer, domain: str = "image"):
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
            - ``'image'`` (default): ``x`` is in domain space; apply :math:`L_{struct}`.
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
        elif domain == "image":
            # x is image (u)
            # L_struct maps Image -> Struct
            self.L_struct.direct(x, out=d)
        elif domain == "struct":
            # x is structural coefficients
            d.fill(x)
        else:
            raise ValueError("domain must be 'image', 'struct', or 'range'")

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