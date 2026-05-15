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

        \min_u \|A u - b\|_2^2

    More generally, LSQR can be applied to a **non-standard L2 regularisation problem**:

    .. math::

        \min_u \|A u - b\|_2^2 + \alpha^2 \|L u\|_2^2,

    where :math:`L=L_{\text{norm}} L_{\text{struct}}` is a instance of the
    `IRLSRegularisationOperator` class.

    With assumptions that :math:`L` is full rank and has an inverse, the problem 
    can be transformed into a standard tikhonov form in the structure space of 
    :math:`x = L u`

    .. math::

        \min_x \|K x - b\|_2^2 + \alpha^2 \| x \|_2^2,

    where :math:`K = A L^{-1}` is the effective operator mapping from the structure 
    space to the data space.

    The solution to the original problem is recovered via

    .. math::

        u(\alpha) = L^{-1} x.

    The operator :math:`K = A L^{-1}` is handled via the `GLSQROperator` class.

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
        the desired structural properties of the solution and has an inverse 
        method implemented.

    The Structural Operator must have an `inverse` method implemented.

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
        xtol: float = 0.1, # Relative tolerance for normal equations for IRLS inner loop
        tau: float = 1.0,
        tau_factor: float = 0.1,
        reinitialize_GKB: bool = True,
        max_inner_iterations: int = 50, # Default maximum inner iterations for IRLS
        **kwargs,
    ):
        """
        Initialise the LSQR algorithm.

        Parameters
        ----------
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data (right-hand side of the equation).
        initial : DataContainer, optional
            Initial guess for the solution. If not provided, a zero-initialised container is used.
        regalpha : float, optional
            Non-negative regularisation parameter. If zero, standard LSQR is used.
        reg_norm_type : str, optional
            Type of regularisation norm ('L1' or 'L2'). Default is 'L2'.
        struct_operator : Operator, optional
            Regularisation operator :math:`L`. If not provided, defaults to IdentityOperator.
        tau : float, optional
            Small positive parameter for L1 regularisation.
        xtol : float, optional
            Relative tolerance for normal equations for IRLS inner loop. Default is 0.1.
        tau_factor : float, optional
            Factor to decrease tau at each outer iteration for L1 regularisation. Default is 0.1.
        reinitialize_GKB : bool, optional
            Whether to reinitialize the Golub-Kahan Bidiagonalisation (GKB) at each outer iteration for L1 regularisation. Default is True.
        max_inner_iterations : int, optional
            Maximum number of inner iterations for IRLS regularisation. Default is 50.
        """
        super().__init__(**kwargs)

        if initial is None:
            initial = operator.domain_geometry().allocate(0)

        self.regalpha = regalpha
        self.reg_norm_type = reg_norm_type

        # L1-norm specific parameters
        self.tau = tau
        self.tau_factor = tau_factor
        self.reinitialize_GKB = reinitialize_GKB

        # Inner loop parameters for IRLS
        self.xtol = xtol
        self.max_inner_iterations = max_inner_iterations
        self.total_gkb_iterations = 0 # Total GKB iterations across inner loops

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

        Unique Geometries
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

        # 2. Identify Geometries
        self.domain_geom_solution = self.operator.domain_geometry()  # Space of u
        self.range_geom_data = self.operator.range_geometry()  # Space of b
        if struct_operator is None:
            self.range_geom_struct = self.domain_geom_solution
        else:
            self.range_geom_struct = struct_operator.range_geometry()


        # 4. Problem sizes (using safe shape product)
        self.domain_size = int(np.prod(self.domain_geom_solution.shape))
        self.data_size = int(np.prod(self.range_geom_data.shape))

        # 5. Allocate variables in their correct spaces
        # Structure Space (Vector x, Search directions v, d)
        self.x = self.range_geom_struct.allocate(0)
        self.v = self.range_geom_struct.allocate(0)
        self.d = self.range_geom_struct.allocate(0)
        
        # Vector u lives in Data Space (Residuals u)
        self.u = self.range_geom_data.allocate(0)

        # Temporary Buffers
        self.tmp_range_data = self.range_geom_data.allocate(0) 
        self.tmp_range_struct = self.range_geom_struct.allocate(0)
        self.tmp_domain = self.domain_geom_solution.allocate(0)

        # Initialize GLSQR Operator
        self.glsqr_operator = GLSQROperator(
            domain_geometry=self.domain_geom_solution,
            range_geometry=self.range_geom_struct,
            # Operators and Norm
            operator=operator, # Forward operator A
            struct_operator=struct_operator,
            norm_type=self.reg_norm_type,
            # Buffers
            tmp_range=self.tmp_range_data,
            tmp_domain=self.tmp_domain,
            tmp_range_struct=self.tmp_range_struct,
            # L1-norm parameters
            tau=self.tau,
            tau_factor=self.tau_factor,
        ) 

        # Initialise Golub-Kahan bidiagonalisation (GKB)
        self._initialize_GKB()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    def update(self):
        """Perform a single LSQR iteration of GLSQR with optional IRLS for L1."""
        if self.reg_norm_type.upper() == "L1":
            # IRLS requires inner iterations for L1 norm.
            self._run_irls_inner_loop()
        else:
            # Standard GKB step for L2 norm.
            self._GKB_step()

    def _run_irls_inner_loop(self):
        """Encapsulated inner loop for IRLS-style regularisation."""
        # Update weights for next outer iteration
        self.glsqr_operator.reg_operator.update_weights(self.x, domain="range")

        # Reset GKB for the new weights
        self._initialize_GKB()  # Maps initial to weighted space

        # Inner Loop
        for inner_it in range(self.max_inner_iterations):
            self._GKB_step()
            if self._check_inner_stop(inner_it):
                break

        self.total_gkb_iterations += inner_it + 1

        if not self.reinitialize_GKB:
            self.initial = self.get_output()
    
    def _bidiag_update(self, input_vec, target_vec, op_func, shift_vec, scalar, buffer1):
        """
        Performs: target = (Op(input) - scalar * shift) / norm
        buffer1: Used to store Op(input). Must match Range of Op.
        buffer2: Used as internal workspace for Op (e.g. Solution Space for L_inv).
        kwargs:  Extra buffers passed to op_func
        """
        # Apply Operator: buffer1 = Op(input_vec) using buffer2 as workspace
        op_func(input_vec, out=buffer1)

        # Combine: buffer1 - scalar * shift_vec -> target_vec
        buffer1.sapyb(1.0, shift_vec, -scalar, out=target_vec)

        norm = target_vec.norm()
        if norm > 0:
            target_vec.divide(norm, out=target_vec)
        return norm

    def _initialize_GKB(self):
        """
        Golub-Kahan Bidiagonalisation (GKB)

        Initialise the GKB process for GLSQR with weighting.
        """
        # 1. Map initial guess to structure space
        self.glsqr_operator.reg_operator.direct(self.initial, out=self.x)

        # 2. u = (b - Kx) / beta, beta is norm of numerator
        # K maps structure -> Data. Kx requires tmp_domain for L_inv application.
        self.beta = self._bidiag_update(
            self.x, self.u, self.glsqr_operator.direct, self.data, -1.0, self.tmp_range_data
        )

        # 3. v = (K*u - 0) / alpha, alpha is norm of numerator
        # K* maps Data -> structure.
        # We use tmp_range_struct for the output of K*u
        self.alpha = self._bidiag_update(
            self.u, self.v, self.glsqr_operator.adjoint, self.v, 0.0, self.tmp_range_struct
        )

        # 4. Initialize scalars and search direction
        self.rhobar, self.phibar = self.alpha, self.beta
        self.normr = self.beta
        self.beta0 = self.beta
        self.res2 = 0.0
        self.d = self.v.copy()

    def _GKB_step(self):
        """single iteration of GKB"""
        # 1. Update u: u = (Kv - alpha*u) / beta
        # Input v is in Structure Space. Output u is in Data Space.
        self.beta = self._bidiag_update(
            self.v, self.u, self.glsqr_operator.direct, self.u, self.alpha, 
            self.tmp_range_data
        )

        # 2. Update v: v = (K*u - beta*v) / alpha
        # Input u is in Data Space. Output v is in Structure Space.
        self.alpha = self._bidiag_update(
            self.u, self.v, self.glsqr_operator.adjoint, self.v, self.beta, 
            self.tmp_range_struct
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
        return self.glsqr_operator.reg_operator.inverse(self.x)