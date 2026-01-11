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

log = logging.getLogger(__name__)


class GLSQROperator(LinearOperator):
    r"""`GLSQROperator`: :math:`L`

                   :math:`X` : domain
                   :math:`Y` : range

    Handles the transformation of non-standard Tikhonov regularization problems into
    standard form for use with the `GLSQR` algorithm allowing for flexible regularisation
    structures and norms.

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
    adapt_tau: bool, optional
        If True, adapt tau during weight updates. Default is True.
    tau_mode: str, optional
        Mode for adapting tau. Options are 'chartrand' (default) and only implemented mode.
    tau_factor: float, optional
        Factor for adapting tau. Default is 0.9.
    tau_min: float, optional
        Minimum value for tau during adaptation. Default is 1e-6.
    """

    def __init__(
        self,
        operator,
        domain_geometry,
        range_geometry=None,
        struct_operator=None,
        norm_type: str = "L2",
        domain_size: int = None,
        tau: float = 1,
        adapt_tau: bool = True,
        tau_mode: str = "factor",
        tau_factor: float = 0.1,
        tau_min: float = 1e-8,
    ):
        # Store domain size and operator
        self.domain_size = domain_size
        self.operator = operator

        # Parameters for IRLS L1 regularisation
        self.tau = tau
        self.adapt_tau = adapt_tau

        # Parameters for tau adaptation
        self.tau_mode = tau_mode
        self.tau_factor = tau_factor
        self.tau_min = tau_min

        # Validate tau adaptation parameters
        if self.tau <= 0:
            raise ValueError("tau must be positive.")
        if self.tau_mode not in ["factor", "solution_norm"]:
            raise ValueError(
                f"Unknown tau_mode '{self.tau_mode}'. Supported modes are 'factor' and 'solution_norm'."
            )
        if tau_factor <= 0 or tau_factor >= 1:
            raise ValueError("tau_factor must be in the interval (0, 1).")
        if tau_min <= 0 or tau_min >= tau:
            raise ValueError("tau_min must be positive and less than tau.")

        # Set structural operator
        if struct_operator is not None:
            self.L_struct = struct_operator
        else:
            self.L_struct = IdentityOperator(domain_geometry)

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
            # FIX: Pass an allocated DataContainer, not the geometry itself
            initial_weights = range_geometry.allocate(self.tau**-0.5)
            self.L_norm = DiagonalOperator(initial_weights)
        else:
            raise ValueError(f"Unknown norm_type '{self.norm_type}'")

        self._null_correction_vector = None  # Cached vector for G_A_dagger
        self._is_gradient_l2 = (
            self.norm_type == "L2"
            and isinstance(self.L_struct, GradientOperator)
        )
        if  (self.norm_type == "L1" and isinstance(self.L_struct, GradientOperator)):
            raise NotImplementedError(
                "L1 norm with Gradient structural operator is not implemented."
            )

        super(GLSQROperator, self).__init__(
            domain_geometry=domain_geometry, range_geometry=range_geometry
        )

    def direct_A_L_inv(self, x, out, tmp_domain):
        """Apply effective operator K = A L_inv"""
        # buffer must be in domain_geometry
        self.inverse(x, out=tmp_domain, tmp_range=out)
        return self.operator.direct(tmp_domain, out=out)

    def adjoint_A_L_inv(self, u, out, tmp_domain):
        """Apply effective adjoint K* = L_inv* A_adj"""
        self.operator.adjoint(u, out=tmp_domain)
        return self.inverse_adjoint(tmp_domain, out=out, tmp_range=tmp_domain)

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

    def adjoint(self, x, out=None):
        r"""Returns the adjoint :math:`L^*(x)=L_{\text{struct}}^*(L_{\text{norm}}^*(x))`

        Parameters
        ----------
        x : DataContainer or BlockDataContainer
            Input data
        out : DataContainer or BlockDataContainer, optional
            If out is not None the output of the Operator will be filled in out, otherwise a new object is instantiated and returned. The default is None.

        Returns
        -------
        DataContainer or BlockDataContainer
            :math:`\mathrm{Id}^*(x)=L_{\text{struct}}^*(L_{\text{norm}}^*(x))`

        """
        if out is None:
            temp = self.L_norm.adjoint(x)
            return self.L_struct.adjoint(temp)
        else:
            self.L_norm.adjoint(x, out=out)
            return self.L_struct.adjoint(out, out=out)
    
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
        self._null_correction_vector = tmp_domain / (self.domain_size * norm_Aw_sq)

    def inverse(self, x, out, tmp_range=None, add_nullspace_correction=False):
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
        if not self._is_gradient_l2:
            self.L_norm.inverse(x, out=out)
            return self.L_struct.inverse(out, out=out)
        
        # 1. Ensure v and norms are cached
        if self._null_correction_vector is None:
            self._precompute_GA_vectors(out, tmp_range)

        # 2. y = G_dagger * x
        self.L_struct.inverse(x, out=out)

        # 3. s = mean(v^T * y)
        # In CIL/NumPy: v.dot(y) / N
        s = self._null_correction_vector.dot(out) / self.domain_size

        # 4. G_A_dagger * x = y - s * e
        # Since e is the vector of ones, we just subtract the scalar s
        out.subtract(s, out=out)

        # 5. Null-space correction (if requested)
        if add_nullspace_correction:
            mean_out = out.sum() / self.domain_size
            out.add(mean_out, out=out)
        
        return out
        
    def inverse_adjoint(self, x, out, tmp_range=None):
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
            self.L_norm.inverse_adjoint(x, out=out)
            return self.L_struct.inverse_adjoint(out, out=out)

        # 1. Ensure v is cached
        if self._null_correction_vector is None:
            self._precompute_GA_vectors(out, tmp_range)

        # 2. mu = mean(x)
        mu = x.sum() / self.domain_size

        # 3. x' = x - mu * v
        x.subtract(self._null_correction_vector * mu, out=out)

        # 4. Result = (G_dagger)^T * x'
        self.L_struct.inverse_adjoint(out, out=out)
        
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
            warnings.warn(
                "update_weights called but norm_type is not 'L1'. No update performed."
            )
            return

        d = self.L_norm.diagonal

        if domain == "range":
            # x is weighted coefficients: L_norm(L_struct(image))
            # Remove weights to get back to structural space
            self.L_norm.inverse(x, out=d)
        elif domain == "image":
            # x is an image: apply L_struct to get structural space
            self.L_struct.direct(x, out=d)
        elif domain == "struct":
            # x is already in structural space: just copy it
            d = x.copy()
        else:
            raise ValueError("domain must be 'image', 'struct', or 'range'")

        # Adapt Tau whilst solution in L_norm space
        if self.adapt_tau:
            self._adapt_tau()

        # # Update weights w = (x^2 + \tau^2)^{-1/4}
        d.power(2, out=d)  # x^2
        d.add(self.tau**2, out=d)  # x^2 + tau^2
        d.power(-0.25, out=d)  # (x^2 + tau^2)^-0.25

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

        if self.tau_mode == "factor":
            # IRLS Strategy:
            # Initialize tau at 1.0
            # Reduce by factor of 10 (tau_factor=0.1) once inner loop stabilizes
            self.tau = max(self.tau * self.tau_factor, self.tau_min)
            log.debug("Tau adapted to: %e", self.tau)
        else:
            log.warning(
                "Unknown tau_mode '%s'. No adaptation performed.", self.tau_mode
            )