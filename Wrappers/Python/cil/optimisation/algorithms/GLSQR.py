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
# CIL Developers and contributers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
from cil.framework import DataContainer
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.operators import GLSQROperator
import numpy as np
import logging

log = logging.getLogger(__name__)


class GLSQR(Algorithm):
    r"""
    Generalised Least Squares with QR factorisation (GLSQR) algorithm.

    The GLSQR algorithm is used to solve large-scale linear systems and least-squares problems,
    particularly when the matrix is sparse or implicitly defined.

    Solves the standard least-squares problem:

    .. math::

        \min_x \|A x - b\|_2^2

    More generally, LSQR can be applied to a **non-standard L2 regularisation problem**:

    .. math::

        \min_u \|A u - b\|_2^2 + \alpha^2 \|\tilde{L} u\|_2^2,

    where :math:`\tilde{L}=L_{\text{norm}} L_{\text{struct}}`.

    Without assumptions on the structure of :math:`L_{\text{struct}}`, this problem is equivalent
    to the standard-form Tikhonov problem (Chung and Gazzola, 2024):

    .. math::

        \bar{x}(\alpha)
        = \underset{\bar{x}}{\mathrm{argmin}}
        \;\|A \tilde{L}_A^{\dagger} \bar{x} - \bar{b}\|_2^2
        + \alpha^2 \|\bar{x}\|_2^2,

    where:

    - :math:`\tilde{L}_A^{\dagger}
        = \left(I - (A(I - \tilde{L}^\dagger \tilde{L}))^\dagger A\right) \tilde{L}^\dagger`
    is the :math:`A`-weighted generalised inverse of :math:`\tilde{L}`.

    - The solution to the original problem is recovered via

    .. math::

        u(\alpha) = \tilde{L}_A^{\dagger} \bar{x}(\alpha) + u_0^{\tilde{L}},

    where :math:`u_0^{\tilde{L}}` is the component of :math:`u` in the null space of :math:`\tilde{L}`.

    - The modified right-hand side is

    .. math::

        \bar{b} = b - u_0^{\tilde{L}}.

    The operator :math:`\tilde{L}` is handled via the `GLSQROperator` class.

    The Norm Operator :math:`L_{\text{norm}}`
    ------------------------------------------
    The choice of :math:`L_{\text{norm}}` depends on the desired regular

    - **L2-norm:** :math:`L_{\text{norm}} = I`, giving the classic Tikhonov-regularised form.

    - **L1-norm:** :math:`L_{\text{norm}}` is diagonal iteratively reweighted operator to approximate
        the L1 norm. The weights are updated at each outer iteration based on the current solution
        estimate.

    The Structural Operator :math:`L_{\text{struct}}`
    ------------------------------------------------
    The choice of :math:`L_{\text{struct}}` depends on the desired regularisation structure:

    - **Wavelets:** :math:`L_{\text{struct}}` is a wavelet transform operator.

    - **Finite Differences:** :math:`L_{\text{struct}}` represents finite difference operators
        for gradient-based regularisation (e.g., Total Variation).

    - **General:** :math:`L_{\text{struct}}` can be any linear operator that captures
        the desired structural properties of the solution and has an inverse or pseudo-inverse 
        method implemented.

    The Structural Operator must have an `inverse` method implemented.

    Parameters
    ----------
    operator : Operator
        Linear operator representing the forward model.
    data : DataContainer
        Measured data (right-hand side of the equation).
    initial : DataContainer, optional
        Initial guess for the solution. If not provided, a zero-initialised container is used.
    alpha : float, optional
        Non-negative regularisation parameter. If zero, standard LSQR is used.
    reg_norm_type : str, optional
        Type of regularisation norm ('L1' or 'L2'). Default is 'L2'.
    weight_operator : Operator, optional
        Regularisation operator :math:`L`. If not provided, defaults to IdentityOperator.
    maxinit : int, optional
        Maximum number of inner iterations for L1 regularisation.
    tau : float, optional
        Small positive parameter for L1 regularisation.
    xtol : float, optional
        Solution change tolerance for stopping criteria for L1 regularisation.

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/lsqr/
    """

    def __init__(
        self,
        operator,
        data: DataContainer,
        initial: DataContainer = None,
        reg_norm_type: str = "L2",
        struct_operator=None,
        regalpha: float = 0.0,
        maxinit: int = 20,
        tau: float = 1e-3,
        xtol: float = 1e-3,
        reinitialize_GKB: bool = True,
        store_subspace_history: bool = False,
        **kwargs,
    ):
        """
        Initialise the LSQR algorithm.

        Parameters
        ----------
        initial : DataContainer, optional
            Initial guess for the solution.
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data.
        regalpha : float, optional
            Regularisation parameter. Default is 0 (no regularisation).
        reg_norm_type : str, optional
            Type of regularisation norm ('L1' or 'L2'). Default is 'L2'.
        weight_operator : Operator, optional
            Tikhonov weight operator :math:`L`. The operator must have an inverse or pseudo-inverse
            method implemented.
        maxinit : int, optional
            Maximum number of inner iterations for L1 regularisation.
        tau : float, optional
            Small positive parameter for L1 regularisation.
        atol : float, optional
            Absolute tolerance for stopping criteria for L1 regularisation.
        btol : float, optional
            Relative tolerance for stopping criteria for L1 regularisation.
        xtol : float, optional
            Solution change tolerance for stopping criteria for L1 regularisation.
        """
        super().__init__(**kwargs)

        if initial is None:
            initial = operator.domain_geometry().allocate(0)

        self.regalpha = regalpha
        self.reg_norm_type = reg_norm_type

        # L1-norm specific parameters
        self.maxinit = maxinit
        self.tau = tau
        self.xtol = xtol
        self.reinitialize_GKB = reinitialize_GKB
        self._store_subspace_history = store_subspace_history

        # Initialise the algorithm
        self.set_up(
            initial=initial,
            operator=operator,
            data=data,
            struct_operator=struct_operator,
            **kwargs,
        )

    def set_up(self, initial, operator, data, struct_operator, **kwargs):
        """
        Set up the GLSQR algorithm with the problem definition.

        Parameters
        ----------
        initial : DataContainer
            Initial guess for the solution.
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data.
        struct_operator : Operator
            Structural regularisation operator :math:`L_{\text{struct}}`.
        """
        log.info("%s setting up", self.__class__.__name__)

        # 1. Problem definitions
        self.operator = operator
        self.data = data
        self.initial = initial  # 1 domain

        # 3. Identify Geometries
        # domain_geom: Image Space (u)
        # range_geom: Data Space (b)
        domain_geom = self.operator.domain_geometry()
        range_geom = self.operator.range_geometry()

        # 3. Define the regularisation operator (\tilde{L} = L_{\text{norm}} L_{\text{struct}})
        self.weight_operator = GLSQROperator(
            domain_geometry=domain_geom,
            struct_operator=struct_operator,
            norm_type=self.reg_norm_type,
            adapt_tau=True,
        )
        if self.reg_norm_type.upper() == "L1":
            self.weight_operator.update_weights(self.initial, domain="image")

        # 4. Allocate variables for iterations
        self.x = domain_geom.allocate(0)  # 2 domain
        self.u = range_geom.allocate(0)  # 1 range
        self.v = domain_geom.allocate(0)  # 3 domain
        self.d = domain_geom.allocate(0)  # 4 domain
        self.tmp_range = range_geom.allocate(0)  # 2 range
        self.tmp_domain = domain_geom.allocate(0)  # 5 domain

        # Initialise Golub-Kahan bidiagonalisation (GKB)
        self._initialize_GKB()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    def update(self):
        """Perform a single iteration of the GLSQR algorithm."""
        # Perform GLSQR Iteration with optional IRLS regularisation for L1 norm
        self._perform_iteration()

    def _initialize_subspace_history(self):
        """Initialize storage for subspace scalars."""
        # Initialise storage for alpha and beta history
        self.alphavec = [self.alpha]
        self.betavec = [self.beta]

        self.k = 1  # Iteration counter for hybrid LSQR

    def _update_subspace_history(self):
        """Store history of alpha and beta."""
        self.alphavec.append(self.alpha)
        self.betavec.append(self.beta)
        self.k += 1

    def _perform_iteration(self):
        """Perform a single LSQR iteration of GLSQR with optional IRLS for L1."""
        if self.reg_norm_type.upper() == "L1":
            # IRLS requires inner iterations for L1 norm.
            self._run_irls_inner_loop()
        else:
            # Standard GKB step for L2 norm.
            self._GKB_step()

    def _run_irls_inner_loop(self):
        """Encapsulated inner loop for IRLS-style regularisation."""
        # Reset GKB for the new weights
        if self.reinitialize_GKB:
            self._initialize_GKB()  # Maps initial to weighted space

        # Inner Loop
        for inner_it in range(self.maxinit):
            self._GKB_step()

            if self._check_inner_stop(inner_it):
                break

        # Update weights for next outer iteration
        self.weight_operator.update_weights(self.x, domain="range")

    def _bidiag_update(self, input_vec, target_vec, op_func, shift_vec, scalar, buffer):
        """
        Performs the GKB update step:
        target = Op(input_vec) - scalar * shift_vec
        """
        # op_func (K or adjoint K*) writes its result into the provided buffer
        op_func(input_vec, out=buffer)

        # Combine (1.0 * buffer) + (-scalar * shift_vec) -> target_vec
        buffer.sapyb(1.0, shift_vec, -scalar, out=target_vec)

        norm = target_vec.norm()
        if norm > 0:
            target_vec.divide(norm, out=target_vec)
        return norm

    def _apply_K(self, x, out):
        """Apply effective operator K = A * L_inv"""
        # buffer must be in domain_geometry
        self.weight_operator.inverse(x, out=self.tmp_domain)
        return self.operator.direct(self.tmp_domain, out=out)

    def _apply_K_adjoint(self, u, out):
        """Apply effective adjoint K* = L_inv * A_adj"""
        self.operator.adjoint(u, out=self.tmp_domain)
        return self.weight_operator.inverse_adjoint(self.tmp_domain, out=out)

    def _initialize_GKB(self):
        """
        Golub-Kahan Bidiagonalisation (GKB)

        Initialise the GKB process for GLSQR with weighting.
        """
        # 1. Map initial guess to weighted space
        self.weight_operator.direct(self.initial, out=self.x)

        # 2. u = (b - Kx) / beta, beta is norm of numerator
        # _apply_K outputs to Range, so use self.tmp_range
        self.beta = self._bidiag_update(
            self.x, self.u, self._apply_K, self.data, -1.0, self.tmp_range
        )

        # 3. v = (K*u - 0) / alpha, alpha is norm of numerator
        # _apply_K_adjoint outputs to Domain, so use self.tmp_domain
        self.alpha = self._bidiag_update(
            self.u, self.v, self._apply_K_adjoint, self.v, 0.0, self.tmp_domain
        )

        # 4. Initialize scalars and search direction
        self.rhobar, self.phibar = self.alpha, self.beta
        self.normr = self.beta
        self.beta0 = self.beta
        self.res2 = 0.0
        self.d = self.v.copy()

        if self._store_subspace_history:
            self._initialize_subspace_history()

    def _GKB_step(self):
        """single iteration of GKB"""
        # Update u: u = (Kv - alpha*u) / beta, beta is norm of numerator
        # _apply_K outputs to Range, so use self.tmp_range
        self.beta = self._bidiag_update(
            self.v, self.u, self._apply_K, self.u, self.alpha, self.tmp_range
        )

        # Update v: v = (K*u - beta*v) / alpha, alpha is norm of numerator
        # _apply_K_adjoint outputs to Domain, so use self.tmp_domain
        self.alpha = self._bidiag_update(
            self.u, self.v, self._apply_K_adjoint, self.v, self.beta, self.tmp_domain
        )

        # 3. Scalar Updates
        rhobar1 = np.hypot(self.rhobar, self.regalpha)
        psi = (self.regalpha / rhobar1) * self.phibar
        phibar_temp = (self.rhobar / rhobar1) * self.phibar

        rho = np.hypot(rhobar1, self.beta)
        c, s = rhobar1 / rho, self.beta / rho

        # Store coefficients for stopping criteria and updates
        self.step_coeff = (c * phibar_temp) / rho
        self.d_update_coeff = (s * self.alpha) / rho

        # Update class state for next iteration
        self.rhobar = -c * self.alpha
        self.phibar = s * phibar_temp
        self.res2 += psi**2
        self.normr = np.hypot(self.phibar, np.sqrt(self.res2))

        # 4. Vector Updates (Using stored coefficients)
        # x = x + step_coeff * d
        self.x.sapyb(1.0, self.d, self.step_coeff, out=self.x)
        # d = v - d_update_coeff * d
        self.v.sapyb(1.0, self.d, -self.d_update_coeff, out=self.d)

        if self._store_subspace_history:
            self._update_subspace_history()

    def update_objective(self):
        """
        Update the objective function value (residual norm squared).
        """
        if np.isnan(self.normr):
            raise StopIteration()
        self.loss.append(self.normr**2)

    def _check_inner_stop(self, inner_it):
        """Check inner stopping criteria for IRLS"""
        # Current image norm
        xnorm = self.x.norm()

        # Tolerances scaled by initial residual beta0
        step_tol = self.xtol * (xnorm + 1.0)

        # Calculate step norm: ||step_coeff * d||
        step_norm = abs(self.step_coeff) * self.d.norm()

        log.debug(
            "Inner It %d: step-tol: %.2e",
            inner_it,
            (step_norm - step_tol),
        )

        if step_norm <= step_tol:
            log.debug("Stopping Criteria: small relative step.")
            return True

        return False

    def get_output(self):
        r"""Returns the current solution.

        The solution is mapped back to the image space via the inverse of the weight operator.

        Returns
        -------
        DataContainer
            The current solution

        """
        # Map back to original space
        return self.weight_operator.inverse(self.x)
