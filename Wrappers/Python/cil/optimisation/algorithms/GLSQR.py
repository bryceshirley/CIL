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
        the desired structural properties of the solution. 

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
        struct_operator=None,
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
            Tikhonov weight operator :math:`L`. The operator must have an inverse or pseudo-inverse
            method implemented. 
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
            struct_operator=struct_operator,
        )

    def set_up(self, initial, operator, data, struct_operator):
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
        self.weight_operator = GLSQROperator(domain_geometry=domain_geom,
                                             range_geometry=range_geom,
                                             struct_operator=struct_operator,
                                             norm_type=self.reg_norm_type,
                                             adapt_tau=True)
        if self.reg_norm_type.upper() == "L1":
            self.weight_operator.update_weights(self.initial)

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
        # # Reset GKB for the new weights
        # self._initialize_GKB()  # Maps initial to weighted space

        # Inner Loop
        for inner_it in range(self.maxinit):
            self._GKB_step()

            if self._check_inner_stop(inner_it):
                break

        # Update weights for next outer iteration
        self.weight_operator.update_weights(self.x, in_transform_domain=True)

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

        The solution is mapped back to the image space via the inverse of the weight operator.

        Returns
        -------
        DataContainer
            The current solution

        """
        # Map back to original space
        return self.weight_operator.inverse(self.x)
