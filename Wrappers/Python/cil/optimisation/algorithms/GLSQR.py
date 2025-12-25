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
from cil.optimisation.operators import DiagonalOperator, IdentityOperator

import numpy as np
import logging
import math

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

        \min_u \|A u - b\|_2^2 + \alpha^2 \|L u\|_2^2,

    where :math:`L` is a (possibly non-diagonal) linear operator acting on the unknown
    :math:`u` (e.g., finite-difference gradient, Laplacian, wavelet transform, or
    other structured operators).

    Without assumptions on the structure of :math:`L`, this problem is equivalent
    to the standard-form Tikhonov problem (Chung and Gazzola, 2024):

    .. math::

        \bar{x}(\alpha)
        = \underset{\bar{x}}{\mathrm{argmin}}
        \;\|A L_A^{\dagger} \bar{x} - \bar{b}\|_2^2
        + \alpha^2 \|\bar{x}\|_2^2,

    where:

    - :math:`L_A^{\dagger}
        = \left(I - (A(I - L^\dagger L))^\dagger A\right) L^\dagger`
    is the :math:`A`-weighted generalised inverse of :math:`L`.

    - The solution to the original problem is recovered via

    .. math::

        u(\alpha) = L_A^{\dagger} \bar{x}(\alpha) + u_0^L,

    where :math:`u_0^L` is the component of :math:`u` in the null space of :math:`L`.

    - The modified right-hand side is

    .. math::

        \bar{b} = b - u_0^L.

    Key Points
    ----------

    - **Standard GLSQR:** :math:`L = I`, giving the classic Tikhonov-regularised form
    :math:`\|A x - b\|_2^2 + \alpha^2 \|x\|_2^2`.

    - **Wavelet GLSQR:** :math:`L` is a wavelet transform operator. The inverse of :math:`L`
    is then the corresponding inverse wavelet transform.

    - **L1-weighted GLSQR:** :math:`L` is diagonal (e.g., IRLS schemes for
    approximate L1 regularisation). The inverse of :math:`L` is then the elementwise
    reciprocal of its diagonal entries.

    - **General weighted GLSQR:** :math:`L` may represent gradient operators,
    differential operators, or transforms (e.g., finite differences, wavelets).
    In such cases, applying the weights requires using :math:`L`'s inverse or
    pseudo-inverse rather than simple elementwise scaling.


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
    maxoutit : int, optional
            Maximum number of outer iterations. Default is the size of the domain.
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
        weight_operator=None,
        regalpha: float = 0.0,
        maxoutit: int = 50,
        maxinit: int = 20,
        tau: float = 1e-3,
        atol: float = 1e-1,
        btol: float = 1e-1,
        xtol: float = 1e-1,
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
            Regularisation operator :math:`L`. The operator must have an inverse or pseudo-inverse
            method implemented. If no weight_operator is provided, defaults to IdentityOperator
            for 'L2' norm and DiagonalOperator for 'L1' norm.
        maxoutit : int, optional
            Maximum number of outer iterations. Default is the size of the domain.
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
        self.maxoutit = maxoutit

        # L1-norm specific parameters
        self.maxinit = maxinit
        self.tau = tau
        self.atol = atol
        self.btol = btol
        self.xtol = xtol

        # Initialise the algorithm
        self.set_up(
            initial=initial,
            operator=operator,
            data=data,
            weight_operator=weight_operator,
        )

    def set_up(self, initial, operator, data, weight_operator):
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

        # 3. Define the weight operator (L)
        self._set_up_weight_operator(weight_operator, domain_geom)

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

    def _set_up_weight_operator(self, weight_operator, geom):
        """
        Set up the weight operator :math:`L` for regularisation.
        Parameters
        ----------
        weight_operator : Operator, optional
            Regularisation operator :math:`L`. If not provided, defaults to IdentityOperator
            for 'L2' norm and DiagonalOperator for 'L1' norm.
        geom : Geometry
            Geometry of the domain space for the weight operator.
        Raises
        -------
        ValueError
            If the provided weight_operator does not have an 'inverse' method.
        ValueError
            If an unknown reg_norm_type is specified.
        """

        # Select weight operator
        if weight_operator is not None:
            self.weight_operator = weight_operator
        else:
            # Select default weight operator based on reg_norm_type
            if self.reg_norm_type.upper() == "L2":
                self.weight_operator = IdentityOperator(geom)
            elif self.reg_norm_type.upper() == "L1":
                self.weight_operator = DiagonalOperator(geom)
                self._update_irls_weights(self.initial)
            else:
                raise ValueError(
                    f"Unknown reg_norm_type '{self.reg_norm_type}'. Supported types are 'L1' and 'L2'."
                )

        # Validate that the weight operator has an inverse method
        if not hasattr(self.weight_operator, "inverse"):
            raise ValueError(
                "The provided weight_operator must have an 'inverse' method implemented."
            )

    def _update_irls_weights(self, x: DataContainer):
        """
        Update DiagonalOperator weights for IRLS L1 regularisation.

        .. math::
            w = (x^2 + \tau^2)^{-1/4}

        Where :math:`\tau` is a small positive parameter to avoid singularities.

        Parameters
        ----------
        x : DataContainer
            Current solution estimate.
        """
        x.power(2, out=self.weight_operator.diagonal) 
        self.weight_operator.diagonal.add(self.tau**2, out=self.weight_operator.diagonal)
        self.weight_operator.diagonal.power(-0.25, out=self.weight_operator.diagonal)

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
        self._initialize_GKB()  # Maps initial to weighted space

        # Inner Loop
        for inner_it in range(self.maxinit):
            self._GKB_step()

            if self._check_inner_stop(inner_it):
                break

        # Transform back to original space
        self.weight_operator.inverse(self.x, out=self.x)

        # Update weights for next outer iteration
        self._update_irls_weights(self.x)

    def _initialize_GKB(self):
        """
        Golub-Kahan Bidiagonalisation (GKB)

        Initialise the GKB process for GLSQR with weighting.
        """
        # 1. Transform initial guess to weighted space: x = L(initial)
        self.weight_operator.direct(self.initial, out=self.x)

        # 2. Compute initial residual in range space: u = A(L_inv(x)) - data
        # We use tmp_domain to hold the intermediate L_inv(x)
        self.weight_operator.inverse(self.x, out=self.tmp_domain)
        self.operator.direct(self.tmp_domain, out=self.u)
        self.u.sapyb(1.0, self.data, -1.0, out=self.u)  # u = Ax - b

        self.beta = self.u.norm()
        self.u.divide(self.beta, out=self.u)  # Problem if u is zero

        # 3. Compute first vector in domain space: v = L_inv(A_adj(u))
        self.operator.adjoint(self.u, out=self.tmp_range)
        self.weight_operator.inverse(self.tmp_range, out=self.v)

        self.alpha = self.v.norm()
        self.v.divide(self.alpha, out=self.v)

        # 4. Initialize scalars, search direction and residuals
        self.rhobar = self.alpha
        self.phibar = self.beta
        self.normr = self.beta
        self.beta0 = self.beta
        self.res2 = 0.0

        # Copy v into d (initial search direction) without re-allocating
        self.v.copy(out=self.d)

    def _GKB_step(self):
        """single iteration of GKB"""
        # Update u in GKB
        self.weight_operator.inverse(self.v, out=self.tmp_domain)
        self.operator.direct(self.tmp_domain, out=self.tmp_range)
        self.tmp_range.sapyb(1.0, self.u, -self.alpha, out=self.u)
        self.beta = self.u.norm()
        self.u.divide(self.beta, out=self.u)

        # Update v in GKB
        self.operator.adjoint(self.u, out=self.tmp_range)
        self.weight_operator.inverse(self.tmp_range, out=self.tmp_domain)
        self.v.sapyb(-self.beta, self.tmp_domain, 1.0, out=self.v)
        self.alpha = self.v.norm()
        self.v.divide(self.alpha, out=self.v)

        # Eliminate diagonal from regularisation
        if self.regalpha == 0.0:  # No regularisation
            rhobar1 = self.rhobar
            psi = 0
        else:
            rhobar1 = np.sqrt(self.rhobar * self.rhobar + self.regalpha**2)
            c1 = self.rhobar / rhobar1
            s1 = self.regalpha / rhobar1
            psi = s1 * self.phibar
            self.phibar = c1 * self.phibar

        # Eliminate lower bidiagonal part
        rho = np.sqrt(rhobar1**2 + self.beta**2)
        c = rhobar1 / rho
        s = self.beta / rho
        theta = s * self.alpha
        self.rhobar = -c * self.alpha
        phi = c * self.phibar
        self.phibar = s * self.phibar
        self.d_update_coeff = theta / rho
        self.step_coeff = phi / rho

        # Estimate residual norm
        self.res2 += psi**2
        self.normr = np.sqrt(self.phibar**2 + self.res2)

        # 1. Update Solution: x = x + step_coeff * d
        # (1 * x) + (step_coeff * d)
        self.x.sapyb(1.0, self.d, self.step_coeff, out=self.x)

        # 2. Update Search Direction: d = v - d_update_coeff * d
        # (1 * v) + (-d_update_coeff * d)
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

        # Use appropriate tolerances: rel residual uses atol, projected gradient uses btol
        rel_res_tol = self.atol * self.beta0
        proj_grad_tol = self.btol * self.beta0
        step_tol = self.xtol * (xnorm + 1.0)

        step_norm = abs(self.step_coeff) * self.d.norm()

        log.debug(
            "Inner It %d: normr-tol: %.2e, phibar-tol: %.2e, step-tol: %.2e",
            inner_it,
            (self.normr - rel_res_tol),
            (abs(self.phibar) - proj_grad_tol),
            (step_norm - step_tol),
        )
        # 1) Relative residual small: Compare current residual with initial
        if self.normr <= rel_res_tol:
            log.debug("Stopping Criteria: relative residual.")
            return True

        # 2) Projected gradient small (phibar tracks it)
        if abs(self.phibar) <= proj_grad_tol:
            log.debug("Stopping Criteria: projected gradient.")
            return True

        # 3) Relative step in x small
        if step_norm <= step_tol:
            log.debug("Stopping Criteria: small relative step.")
            return True

        return False

    def get_output(self):
        r"""Returns the current solution.

        For L2 regularisation, the solution is mapped back to the original space
        using the inverse of the weight operator. For L1 regularisation, the
        solution is already in the original space.

        Returns
        -------
        DataContainer
            The current solution

        """
        # Map back to original space
        if self.reg_norm_type.upper() == "L2":
            return self.weight_operator.inverse(self.x)
        elif self.reg_norm_type.upper() == "L1":
            return self.x
