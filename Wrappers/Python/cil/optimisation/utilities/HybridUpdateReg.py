import numpy as np
import logging
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Optional, Tuple

log = logging.getLogger(__name__)


class UpdateRegBase(ABC):
    """Base class for updating regularization parameters in iterative solvers.

    Parameters
    ----------
    tol : float
        Relative tolerance for convergence.
    data_size : int
        Number of elements in the data; corresponds to range of the operator
        (denoted as 'm').
    domain_size : int
        Number of elements in the solution; corresponds to domain of the operator
        (denoted as 'n').
    """

    def __init__(self, data_size: int, domain_size: int, tol: float = 1e-2):
        # Validation for initialization parameters
        if data_size <= 0 or domain_size <= 0:
            raise ValueError(
                f"Dimensions data_size and domain_size must be positive integers. "
                f"Got data_size={data_size}, domain_size={domain_size}"
            )
        if tol <= 0:
            raise ValueError(f"Tolerance must be greater than 0. Got {tol}")

        self.rule_type = "base rule"
        self.m = data_size
        self.n = domain_size
        self.tol = tol

        # History and state variables
        self.regalpha = 0.0
        self.regalpha_history = []  # History of regularization parameters
        self.func_history = []  # History of function values

        self.converged = False
        self.iteration = 0
        self.EPS = 1e-12

        # Bounds Based on SVD of Bk
        self.regalpha_low = 0.0
        self.regalpha_high = None

    def update_regularizationparam(self, Bk, b_norm):
        """Main entry point to update the regularization parameter.

        Parameters
        ----------
        Bk : np.ndarray
            The projection matrix (k+1 x k).
        b_norm : float
            The norm of the initial right-hand side (data) vector.
        """
        # Validation for runtime inputs
        if not isinstance(Bk, np.ndarray):
            raise TypeError(f"Bk must be a numpy.ndarray, got {type(Bk)}")

        k_plus_1, k = Bk.shape
        if k_plus_1 != k + 1:
            raise ValueError(f"Bk must have shape (k+1, k). Got {Bk.shape}")

        if k > self.n or k_plus_1 > self.m:
            raise ValueError(
                f"Krylov subspace size {k} cannot exceed operator dimensions (m={self.m}, n={self.n})"
            )

        if not np.isscalar(b_norm):
            raise TypeError(f"b_norm must be a scalar norm, got {type(b_norm)}")

        # Update iteration count
        self.iteration = Bk.shape[1]

        # Compute SVD and initialize subspace components
        self._initialize_subspace_components(Bk, b_norm)

        # Compute the next regularization parameter
        new_regalpha = self._compute_next_regalpha()

        # Update state if a new regularization parameter is found
        if new_regalpha is not None:
            self.regalpha = new_regalpha

            # Append to history
            self.regalpha_history.append(new_regalpha)
            # Verify func_value is scalar and not None or Nan or infinite
            func_value = self.func(new_regalpha)
            if (
                np.isscalar(func_value)
                and func_value is not None
                and not np.isnan(func_value)
                and not np.isinf(func_value)
            ):
                self.func_history.append(func_value)

            self._run_convergence_checks()
        self._log_status()

    def _residual_filter(self, reg):
        """
        Computes the residual filter factors for a given regularization parameter.

        The residual filter, often denoted as :math:`f_r(s,\alpha)`, dampens the
        influence of small singular values on the residual. For Tikhonov
        regularization, it is defined as:

        .. math::

            f_r(s,\alpha) = \frac{\alpha^2}{s^2 + \alpha^2}

        Where:
        * :math:`\alpha` is the regularization parameter (``reg``).
        * :math:`s` are the singular values of the system (``Sb``).
        """
        return (reg**2) / (self.Sbsq + reg**2)

    def _solution_filter(self, reg):
        """
        Computes the solution filter factors for a given regularization parameter.

        The solution filter, or filter function :math:`f_x`, scales the
        singular values to provide a smoothed pseudo-inverse solution:

        .. math::

            f_x(s,\alpha) = \frac{s}{s^2 + \alpha^2}

        This filter ensures that as :math:`s \to 0`, the solution does not
        blow up, effectively suppressing noise in the inversion process.
        """
        return self.Sb / (self.Sbsq + reg**2)

    def _projected_residual_norm_sq(self, reg):
        """
        Computes the squared projected residual norm :math:`\|r_\alpha\|_2^2`.

        The residual norm in the :math:`k`-th Krylov subspace is given by:
        .. math::
            \rho(\alpha)^2 = \sum_{i=1}^k \left( f_r(s_i,\alpha) \hat{\beta}_i \right)^2
            + \hat{\beta}_{k+1}^2

        Where:
         * :math:`\hat{\beta}_i = u_i^T (\|b\|_2 e_1)` represents the :math:`i`-th component of the projected
            right-hand side or data (the projection of the initial residual onto the left singular vectors of the
            projected bidiagonal matrix).
         * :math:`f_r(s_i,\alpha)` are the residual filter factors defined in method ``_residual_filter``.

        Parameters
        ----------
        reg : float
            The regularization parameter :math:`\alpha`.

        Returns
        -------
        float
            Squared norm of the residual.
        """
        # Residual filter
        fr = self._residual_filter(reg)

        # Contribution from the Tikhonov-filtered components within the subspace
        residual_1 = np.sum((fr * (self.b_norm * self.u1)) ** 2)

        # Contribution from non-filtered component
        residual_2 = (self.b_norm * self.u1_tail) ** 2

        return residual_1 + residual_2

    def _projected_solution_norm_sq(self, reg):
        """
        Computes the squared projected solution norm :math:`\|x_\alpha\|_2^2`.

        The solution norm in the :math:`k`-th Krylov subspace is given by:
        .. math::
            \eta(\alpha)^2 = \sum_{i=1}^k \left( f_x(s_i,\alpha) \hat{\beta}_i \right)^2

        Where:
         * :math:`\hat{\beta}_i = u_i^T (\|b\|_2 e_1)` represents the :math:`i`-th component of the projected
            right-hand side or data (the projection of the initial residual onto the left singular vectors of the
            projected bidiagonal matrix).
         * :math:`f_x(s_i,\alpha)` are the solution filter factors defined in method ``_solution_filter``.

        Parameters
        ----------
        reg : float
            The regularization parameter :math:`\alpha`.

        Returns
        -------
        float
            Squared norm of the solution.
        """
        # Solution filter
        fx = self._solution_filter(reg)

        # Only the first k components contribute to the solution norm
        residual = np.sum(np.square((self.b_norm * self.u1) * fx))

        return residual

    def _projected_norm_first_derivatives(self, reg):
        r"""Compute analytical squared projected norms and their 1st/2nd derivatives.

        **Variable Naming Convention:**

        In the code, suffixes are used to represent derivatives with respect
        to the regularization parameter :math:`\alpha`:

        * **_p**: First derivative, :math:`\frac{d}{d\alpha}` (e.g., ``R2_p`` is :math:`(R^2)'`).
        * **_pp**: Second derivative, :math:`\frac{d^2}{d\alpha^2}` (e.g., ``R2_pp`` is :math:`(R^2)''`).

        Returns
        -------
        tuple
            (R2_p, X2_p) if `second_derivatives` is False, else (R2_p, R2_pp, X2_p, X2_pp)
        """
        fr = self._residual_filter(reg)
        fx = self._solution_filter(reg)
        fr_p, fx_p = self._filter_first_derivatives(reg)

        # First and Second Derivatives of Squared Projected Norms
        b_norm_sq = self.b_norm * self.b_norm
        u1_sq = self.u1 * self.u1

        R2_p = 2 * b_norm_sq * np.sum(u1_sq * fr * fr_p)
        X2_p = 2 * b_norm_sq * np.sum(u1_sq * fx * fx_p)

        return R2_p, X2_p
    
    def _projected_norm_second_derivatives(self, reg):
        r"""Compute analytical squared projected norms and their 2nd derivatives.

        **Variable Naming Convention:**

        In the code, suffixes are used to represent derivatives with respect
        to the regularization parameter :math:`\alpha`:

        * **_pp**: Second derivative, :math:`\frac{d^2}{d\alpha^2}` (e.g., ``R2_pp`` is :math:`(R^2)''`).

        Returns
        -------
        tuple
            (R2_pp, X2_pp)
        """
        # SVD-based filter factors and their second derivatives
        fr = self._residual_filter(reg)
        fx = self._solution_filter(reg)
        fr_p, fx_p = self._filter_first_derivatives(reg)
        fr_pp, fx_pp = self._filter_second_derivatives(reg)

        # Second Derivatives of Squared Projected Norms
        b_norm_sq = self.b_norm * self.b_norm
        u1_sq = self.u1 * self.u1

        R2_pp = 2 * b_norm_sq * np.sum(u1_sq * (fr_p**2 + fr * fr_pp))
        X2_pp = 2 * b_norm_sq * np.sum(u1_sq * (fx_p**2 + fx * fx_pp))

        return R2_pp, X2_pp
    
    def _projected_norm_third_derivatives(self, reg):
        r"""
        Compute analytical third derivatives of squared projected residual
        and solution norms.

        Returns
        -------
        tuple
            (R2_ppp, X2_ppp)
        """
        fr = self._residual_filter(reg)
        fx = self._solution_filter(reg)
        fr_p, fx_p = self._filter_first_derivatives(reg)
        fr_pp, fx_pp = self._filter_second_derivatives(reg)
        fr_ppp, fx_ppp = self._filter_third_derivatives(reg)

        b_norm_sq = self.b_norm**2
        u1_sq = self.u1**2

        R2_ppp = 2 * b_norm_sq * np.sum(u1_sq * (3 * fr_p * fr_pp + fr * fr_ppp))
        X2_ppp = 2 * b_norm_sq * np.sum(u1_sq * (3 * fx_p * fx_pp + fx * fx_ppp))

        return R2_ppp, X2_ppp

    
    def _filter_first_derivatives(self, reg):
        r"""
        Compute first derivatives of Tikhonov filter factors.

        The filters are defined as defined in :meth:`_residual_filter` and
        :meth:`_solution_filter`.

        The derivatives with respect to :math:`\alpha` are:

        .. math::
           f_r' = \frac{2\alpha\sigma^2}{(\sigma^2 + \alpha^2)^2}, \quad
           f_r'' = \frac{2\sigma^2(\sigma^2 - 3\alpha^2)}{(\sigma^2 + \alpha^2)^3}

        .. math::
           f_x' = \frac{-2\alpha\sigma}{(\sigma^2 + \alpha^2)^2}, \quad
           f_x'' = \frac{2\sigma(3\alpha^2 - \sigma^2)}{(\sigma^2 + \alpha^2)^3}

        In the code, suffixes are used to represent derivatives with respect
        to the regularization parameter :math:`\alpha`:

        * **_p**: First derivative, :math:`\frac{d}{d\alpha}` (e.g., ``R2_p`` is :math:`(R^2)'`).

        Parameters
        ----------
        reg : float
            The regularization parameter :math:`\alpha`.

        Returns
        -------
        tuple
            ( fr_p, fx_p)
        """
        a2 = reg**2
        s2 = self.Sbsq
        denom2 = (s2 + a2)**2

        fr_p = 2 * reg * s2 / denom2
        fx_p = -2 * reg * self.Sb / denom2
        return fr_p, fx_p
    
    def _filter_second_derivatives(self, reg):
        r"""
        Compute second derivatives of Tikhonov filter factors.
        The filters are defined as defined in :meth:`_residual_filter` and
        :meth:`_solution_filter`.

        The second derivatives with respect to :math:`\alpha` are:
        * **_pp**: Second derivative, :math:`\frac{d^2}{d\alpha^2}` (e.g., ``R2_pp`` is :math:`(R^2)''`).
        
        Parameters
        ----------
        reg : float
            The regularization parameter :math:`\alpha`.
        Returns
        -------
        tuple
            (fr_pp, fx_pp)
        """
        a2 = reg**2
        s2 = self.Sbsq
        denom3 = (s2 + a2)**3

        fr_pp = 2 * s2 * (s2 - 3 * a2) / denom3
        fx_pp = 2 * self.Sb * (3 * a2 - s2) / denom3
        return fr_pp, fx_pp

    def _filter_third_derivatives(self, reg):
        r"""
        Compute third derivatives of Tikhonov filter factors.

        Returns
        -------
        tuple
            (fr_ppp, fx_ppp)
        """
        a2 = reg**2
        s2 = self.Sbsq

        fr_ppp = 24 * reg * s2 * (a2 - s2) / (s2 + a2)**4
        fx_ppp = 24 * reg * self.Sb * (s2 - a2) / (s2 + a2)**4

        return fr_ppp, fx_ppp
    


    def _run_convergence_checks(self):
        """Standard convergence check based on the saturation of alpha.

        Uses a relative change formula with a safety epsilon EPS.
        """
        if len(self.regalpha_history) < 2:
            return
        denom = abs(self.regalpha_history[-1]) + self.EPS
        rel_change = abs(self.regalpha_history[-1] - self.regalpha_history[-2]) / denom

        if rel_change < self.tol:
            log.debug(f"Alpha Saturation: Converged at iteration {self.iteration}")
            self.converged = True

    @abstractmethod
    def _compute_next_regalpha(self):
        """Abstract method to compute the next regularization parameter.

        Returns
        -------
        new_regalpha : float
            The updated regularization parameter.
        """
        pass

    @abstractmethod
    def func(self, regalpha):
        """Abstract method to compute the rule-specific function value.

        Parameters
        ----------
        regalpha : float
            The regularization parameter at which to evaluate the function.

        Returns
        -------
        value : float
            The computed function value.
        """
        pass

    def _initialize_subspace_components(self, Bk, b_norm):
        """
        Perform SVD decomposition of the projected operator and initialize
        subspace components for regularization parameter selection.

        The projected right-hand side is decomposed into:
        1. **Head components**: Associated with the $k$ singular values in the
           current Krylov subspace, representing the regularizable signal.
        2. **Tail component**: The $(k+1)$-th component representing the
           residual portion orthogonal to the current subspace, which defines
           the "noise floor" for the L-curve and Discrepancy Principle.

        Parameters
        ----------
        Bk : np.ndarray
            The $(k+1 \times k)$ projection matrix from the bidiagonalization
            process.
        b_norm : float
            The $L_2$ norm of the initial residual/data vector $b$, used to
            scale the projected coefficients.
        """
        # Ub is (k+1 x k+1), Sb is (k,)
        Ub, self.Sb, _ = scipy.linalg.svd(Bk)

        # Store squared singular values and extrema
        self.Sbsq = np.square(self.Sb)
        self.Sbmax = self.Sb[0]
        self.Sbmin = self.Sb[-1]

        # Optimization Upper Bound
        self.regalpha_high = self.Sbmax

        # The first row of Ub corresponds to the projection of the
        # initial residual onto the left singular vectors of Bk.
        # Head: first k elements; Tail: the (k+1)-th element.
        self.u1 = Ub[0, :-1]
        self.u1_tail = Ub[0, -1]

        # Store b_norm
        self.b_norm = b_norm

    def plot_history(self, show_objective=False):
        """
        Utility to plot the history of regularization parameters.

        Parameters
        ----------
        show_objective : bool, optional
            If True, plots both the alpha history and the objective function
            value history in subplots. Default is False.
        """
        # Determine number of subplots
        if show_objective and len(self.func_history) == 0:
            raise ValueError("No function history available to plot.")
        if len(self.regalpha_history) == 0:
            raise ValueError("No regularization parameter history available to plot.")

        # Determine number of subplots
        num_subs = 2 if show_objective else 1

        fig, axes = plt.subplots(num_subs, 1, figsize=(8, 4 * num_subs), sharex=True)

        # Handle axes indexing for 1 vs 2 subplots
        ax_alpha = axes[0] if show_objective else axes

        # Subplot 1: Regularization History
        ax_alpha.plot(
            self.regalpha_history, marker="o", color="tab:blue", label=r"$\alpha$"
        )
        ax_alpha.set_ylabel(r"Regularization $\alpha$")
        ax_alpha.set_yscale("log")
        ax_alpha.set_title(f"{self.rule_type.upper()} Regularization Parameter History")
        ax_alpha.grid(True, which="both", ls="-", alpha=0.5)

        if show_objective:
            # Subplot 2: Objective History
            ax_func = axes[1]
            iters = np.arange(len(self.func_history))
            ax_func.plot(
                iters, self.func_history, marker="x", color="tab:red", linestyle="--"
            )
            ax_func.set_ylabel(f"{self.rule_type.upper()} Function History")
            ax_func.set_xlabel("Iteration")
            ax_func.grid(True, alpha=0.5)
        else:
            ax_alpha.set_xlabel("Iteration")

        plt.tight_layout()
        plt.show()

    def _geometric_grid(
        self,
        regalpha_limits: Optional[Tuple[float, float]] = None,
        num_points: int = 80,
    ):
        r"""
        Generate a geometric grid of regularization parameters and evaluate the objective function
        and find the index of the current regularization parameter.

        Args:
            regalpha_limits (Optional[Tuple[float, float]]):
                The (min, max) range for :math:`\alpha`. Defaults to (self.regalpha_low, self.regalpha_high).
            num_points (int):
                Number of points in the grid. Defaults to 80.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]:
                - regalpha_grid: Array of shape (num_points,) containing :math:`\alpha` values.
                - func_grid: Array of shape (num_points,) containing function values.
                - reg_idx: Index of the current regularization parameter in the grid.

        Raises:
            ValueError: If the limits are negative or if the upper bound is not
                        strictly greater than the lower bound.
        """
        if regalpha_limits is None:
            # Use a safe floor for the lower bound to prevent np.geomspace(0, ...) errors
            low = max(self.regalpha_low, self.EPS)
            high = max(self.regalpha_high, low * 10.0)  # Ensure a valid range
            regalpha_limits = (low, high)

        start, stop = regalpha_limits

        # Geometric space requires strictly positive bounds
        if start <= 0:
            start = self.EPS

        if stop <= start:
            raise ValueError(
                f"Upper bound ({stop}) must be greater than lower bound ({start})."
            )

        # 1. Generate the grid
        regalpha_grid = np.geomspace(start, stop, num_points)

        # 2. Evaluate the function
        # Using np.array for the result for easier downstream slicing/plotting
        func_grid = np.array([self.func(reg) for reg in regalpha_grid])

        # Find index of chosen alpha in grid
        reg_idx = int(np.argmin(np.abs(regalpha_grid - self.regalpha)))

        return regalpha_grid, func_grid, reg_idx

    def plot_function(self, regalpha_limits: Optional[Tuple[float, float]] = None):
        """Utility to plot the rule-specific function over a range of alphas.
        Parameters
        ----------
        regalpha_limits : Optional[Tuple[float, float]]
            Tuple specifying the (min, max) range of regularization parameters to plot.
            If None, defaults to the class's regalpha_low and regalpha_high.
        """
        log.info("Plotting function not implemented")

    def _log_status(self):
        """Logs the current iteration status."""
        val = f"{self.regalpha:.4e}" if self.regalpha else "N/A"
        log.info(
            f"Iteration {self.iteration}: regalpha = {val} [{self.rule_type.upper()}]"
        )


class UpdateRegDiscrep(UpdateRegBase):
    r"""Discrepancy Principle for choosing the regularisation parameter :math:`\alpha`.

    The algorithm finds the root :math:`\alpha` of the discrepancy function:

    .. math:: \phi(\alpha) = \|r_{\alpha}\|_2^2 - \eta^2 = 0

    where :math:`\eta` is the :code:`discrep_noise_level`.

    In the :math:`k`-th Krylov subspace, the residual norm is computed using the SVD
    of the bidiagonal matrix :math:`B_k = U \Sigma V^T`:

    .. math:: \|r_\alpha\|_2^2 = \sum_{i=1}^{k} \left( \frac{\alpha^2}{s_i^2 + \alpha^2} \beta \hat{u}_i \right)^2 + (\beta \hat{u}_{k+1})^2

    where:

    * :math:`s_i` are the singular values of :math:`B_k`.
    * :math:`\beta` is the norm of the initial right-hand side vector :math:`b`.
    * :math:`\hat{u}_i` is the :math:`i`-th element of the first row of :math:`U`.
    * The last term :math:`(\beta \hat{u}_{k+1})^2` is the residual component orthogonal to the current Krylov subspace.

    Parameters
    ----------
    tol : float
        Relative tolerance for convergence of the regularisation parameter.
    data_size : int
        Number of elements in the data; corresponds to range of the operator
        (denoted as 'm').
    domain_size : int
        Number of elements in the solution; corresponds to domain of the operator
        (denoted as 'n').
    noise_level_estimate : float
        The estimated norm of the noise in the right-hand side, :math:`\eta = \|e\|_2`.

    Note
    ----
    The discrepancy principle requires an accurate estimate of the noise level.
    If :math:`\eta` is underestimated, the solution may be under-regularised;
    if overestimated, the solution will be over-smoothed.

    """

    def __init__(
        self,
        data_size: int,
        domain_size: int,
        tol: float = 1e-2,
        noise_level_estimate=0.0,
    ):
        super().__init__(data_size, domain_size, tol)
        self.rule_type = "discrep"
        self.noise_level_estimate = noise_level_estimate

    def _compute_next_regalpha(self):
        """Finds alpha such that ||r_alpha|| = noise_level_estimate using Brent's method.

        Returns
        -------
        regalpha : float
            The updated regularization parameter.
        """
        # Bracket the root: lo=0 (least smoothing), hi=large (most smoothing)
        f_lo, f_hi = self.func(self.regalpha_low), self.func(self.regalpha_high)

        # If f_lo > 0, it means even with zero smoothing (alpha=0),
        # the residual is ALREADY larger than the noise level.
        if f_lo > 0:
            return None

        # If f_hi < 0, it means even with max smoothing,
        # the residual is smaller than the noise level.
        if f_hi < 0:
            return None

        result = scipy.optimize.root_scalar(
            self.func, bracket=[self.regalpha_low, self.regalpha_high], method="brentq"
        )
        if result.converged:
            new_regalpha = result.root
        else:
            new_regalpha = None

        return new_regalpha

    def func(self, regalpha):
        """Discrepancy function: phi(alpha) = ||r_alpha||^2 - noise^2."""
        return self._projected_residual_norm_sq(regalpha) - self.noise_level_estimate**2

    def plot_function(
        self,
        regalpha_limits: Optional[Tuple[float, float]] = None,
        num_points: int = 80,
        filepath: Optional[str] = None,
    ):
        r"""
        Plot the discrepancy function and residual norm over a range of regularization parameters.

        This method creates two subplots:
        1. **Discrepancy Function**: Plots :math:`\phi(\alpha) = \|r_\alpha\|_2^2 - \eta^2`.
           The root is found where this curve crosses zero.
        2. **Residual Norm**: Plots :math:`\|r_\alpha\|_2`. The discrepancy principle
           selects :math:`\alpha` where the residual norm matches the noise level
           estimate :math:`\eta`.

        Args:
            regalpha_limits (Optional[Tuple[float, float]]):
                The (min, max) range for :math:`\alpha`.
                Defaults to (self.regalpha_low, self.regalpha_high).
            num_points (int):
                Number of points in the grid. Defaults to 80.
            filepath (Optional[str]):
                If provided, saves the plot to the specified file path.
        """
        # Generate grid data and evaluate discrepancy function
        regalpha_grid, func_grid, reg_idx = self._geometric_grid(
            regalpha_limits, num_points
        )

        # Calculate residual norms for the second plot
        # ||r|| = sqrt(phi(alpha) + eta^2)
        res_norm_grid = np.sqrt(func_grid + self.noise_level_estimate**2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # --- Subplot 1: Discrepancy Function ---
        ax1.semilogx(regalpha_grid, func_grid, label="Discrepancy $\phi(\\alpha)$")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax1.semilogx(
            regalpha_grid[reg_idx],
            func_grid[reg_idx],
            "ro",
            markersize=8,
            label=rf"$\alpha={self.regalpha:.3e}$",
        )
        ax1.axvline(self.regalpha, color="gray", linestyle=":", alpha=0.5)
        ax1.set_xlabel("Regularisation parameter ($\\alpha$)")
        ax1.set_ylabel("$\|r_\\alpha\|_2^2 - \eta^2$")
        ax1.set_title("Discrepancy Function Root")
        ax1.legend()
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        # --- Subplot 2: Residual Norm vs Noise Level ---
        ax2.loglog(
            regalpha_grid, res_norm_grid, label="Residual Norm $\|r_\\alpha\|_2$"
        )
        # Horizontal line for noise level estimate
        ax2.axhline(
            self.noise_level_estimate,
            color="green",
            linestyle="--",
            label=rf"Noise Level $\eta={self.noise_level_estimate:.2e}$",
        )
        ax2.loglog(regalpha_grid[reg_idx], res_norm_grid[reg_idx], "ro", markersize=8)
        ax2.axvline(self.regalpha, color="gray", linestyle=":", alpha=0.5)
        ax2.set_xlabel("Regularisation parameter ($\\alpha$)")
        ax2.set_ylabel("$\|r_\\alpha\|_2$")
        ax2.set_title("Residual Norm vs. Target Noise")
        ax2.legend()
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath)

        plt.show()
        plt.close()

class UpdateRegGCV(UpdateRegBase):
    r"""Generalized Cross-Validation (GCV) method for choosing the regularisation parameter :math:`\alpha`.

    This rule identifies the optimal :math:`\alpha` by minimizing the GCV function,
    which serves as a proxy for the predictive mean square error. It supports
    standard, weighted, and adaptive-weighted variations.

    The GCV function :math:`G(\alpha)` is defined as:

    .. math:: G(\alpha) = \frac{k \|r_\alpha\|_2^2}{[\text{trace}(I - \omega A_\alpha)]^2}

    where :math:`k` is the current iteration (subspace dimension), :math:`r_\alpha` is
    the projected residual, and :math:`\omega` is a weighting factor.

    - **Standard GCV**: :math:`\omega = 1`.
    - **Weighted GCV**: :math:`\omega` is a fixed scalar (typically :math:`< 1`) to
      correct for the undersmoothing tendency of standard GCV.
    - **Adaptive GCV**: :math:`\omega` is updated at each iteration based on the
      noise level and singular value decay to provide a more robust estimate.

    The minimization is performed using a bounded search within the range
    :math:`[\alpha_{low}, \alpha_{high}]`. Convergence is monitored via the
    stability of the :math:`\hat{G}` functional.

    Parameters
    ----------
    tol : float
        Relative tolerance for convergence of the regularisation parameter.
    data_size : int
        Number of elements in the data; corresponds to range of the operator
        (denoted as 'm').
    domain_size : int
        Number of elements in the solution; corresponds to domain of the operator
        (denoted as 'n').
    gcv_weight : float, optional
        The weighting parameter :math:`\omega`. Defaults to 1.0 (standard GCV).
    adaptive_weight : bool, optional
        If True, ignores `gcv_weight` and computes an adaptive :math:`\omega`
        at each iteration. Defaults to True.
    """

    def __init__(
        self,
        data_size: int,
        domain_size: int,
        tol: float = 1e-2,
        gcv_weight: float = 1.0,  # 1.0 is standard GCV
        adaptive_weight: bool = True,  # Whether to use adaptive weighting
    ):
        super().__init__(data_size, domain_size, tol)
        if adaptive_weight:
            gcv_type = "adaptive-weighted"
        elif gcv_weight == 1.0:
            gcv_type = "standard"
        else:
            gcv_type = "weighted"
        self.rule_type = f"{gcv_type} gcv"
        self.gcv_type = gcv_type
        self.omega = gcv_weight
        self.omega_history = []
        self.Ghat_history = []

    def _compute_next_regalpha(self):
        """
        Minimize the GCV function to find the next regularization parameter.
        Returns
        -------
        regalpha : float
            The updated regularization parameter.
        """
        if self.gcv_type == "adaptive-weighted":
            self.omega = self._adaptive_omega()

        # Grid search in log space to find a good starting point (x0)
        regalpha_grid, func_grid, _ = self._geometric_grid()

        # Find the peak from the grid, avoiding the very first/last points
        # to bypass the boundary plateaus seen in your plots.
        grid_search_idx = np.argmin(func_grid)
        x0 = regalpha_grid[grid_search_idx]

        # Minimize the log of alpha
        res = scipy.optimize.minimize(
            lambda reg: self.func(reg),  # Transform back inside the call
            x0=x0,
            bounds=[(self.regalpha_low, self.regalpha_high)],
            tol=1e-10,
        )

        if res.success and np.isfinite(res.x[0]):
            new_regalpha = res.x[0]  # Return the actual alpha
        else:
            new_regalpha = x0  # Fallback to grid search alpha

        # Update GHat history for convergence monitoring
        if self.iteration > 0:
            self.Ghat_history.append(self.Ghat_func(self.regalpha))
        return new_regalpha

    def _adaptive_omega(self):
        """
        Compute an adaptive GCV weighting parameter omega.

        For early iterations we need small regularization so we solve for
        the weight omega that gives the derivative of the GCV function
        equal to zero.

        For late iterations, projected Bk will inheret more and more of the
        ill-conditioning present in forward operator, at some point when the
        ratio Sbmin/Sbmax is very small, we average the omega values over
        previous iterations to stabilize the estimate.
        """
        # Compute adaptive omega from Gradient of GCV functional
        filt = 1 / (self.Sbmin + self.Sbsq)
        u1_tail_sq = self.u1_tail * self.u1_tail
        num = (
            (self.iteration + 1)
            * self.Sbmin
            * self.Sbmin
            * np.sum(u1_tail_sq * self.Sbsq * np.power(filt, 3))
        )
        denom1 = (
            np.sum(np.power(self.Sbmin, 4) * u1_tail_sq * np.power(filt, 2))
            + u1_tail_sq
        )
        denom2 = np.sum(self.Sbsq * np.power(filt, 2))
        denom3 = (
            self.Sbmin * self.Sbmin * np.sum(u1_tail_sq * self.Sbsq * np.power(filt, 3))
        )
        denom4 = np.sum(self.Sbsq * filt)

        omega = num / (denom1 * denom2 + denom3 * denom4)
        self.omega_history.append(omega)

        # Stabilize omega in late iterations
        if self.Sbmin / self.Sbmax < 1e-6:
            omega = np.mean(self.omega_history)
        return omega

    def _weighted_trace(self, regalpha, constrained_dofs):
        """
        Compute the weighted trace term in the GCV denominator:
        trace(I - omega * A_alpha * A^{#}) = constrained_dofs + sum_i filt_i
        where filt_i = ((1 - omega) * sigma_i^2 + alpha^2) / (sigma_i^2 + alpha^2)
        """
        filt = ((1 - self.omega) * self.Sbsq + regalpha**2) / (self.Sbsq + regalpha**2)
        return np.square(constrained_dofs + np.sum(filt))

    def func(self, regalpha):
        """
        Generalized Cross-Validation (GCV) functional to minimize for
        regularization parameter selection.

        .. math:: G(alpha) = (k * ||r_alpha||^2) / (trace(I - omega * A_alpha))^2

        where:
            - k is the current iteration (subspace dimension)
            - r_alpha is the projected residual
            - omega is the weighting factor
        """
        return (
            self.iteration
            * self._projected_residual_norm_sq(regalpha)
            / self._weighted_trace(regalpha, 1)
        )

    def Ghat_func(self, regalpha):
        """
        Chung, Nagy, and O'Leary (2008) propose a modified GCV functional
        to monitor convergence of the regularization parameter. We define:

        .. math:: \hat{G}(\omega,\alpha_k) = n*||b||^2 * [ \sum_{i=1}^k ( (\alpha_k^2 * u1_i * e_1) / (\sigma_i^2 + \alpha_k^2) )^2 + (u_{k+1} * e_1)^2 ]
                    / [ (m - k) + \sum_{i=1}^k ((1-\omega)\sigma_i^2 + \alpha_k^2) / (\sigma_i^2 + \alpha_k^2) ) ]^2
        """
        return (
            self.n
            * self._projected_residual_norm_sq(regalpha)
            / self._weighted_trace(regalpha, self.m - self.iteration)
        )

    def _run_convergence_checks(self):
        """
        Override base method to include GHat convergence check based on Chung et al. (2008).
        ----------------------------------------------------------------------------------
        1. Calls the base class convergence checks based on regalpha convergence.
        2. Appends the current GHat (math:: \hat{G}(\omega,\alpha_k)) value to history, defined
              in :meth:`Ghat_func`.
        3. Checks for convergence based on two simultaneous criteria:
            a. Convergence: math:: |\hat{G}(k) - \hat{G}(k-1)| / |\hat{G}(1)| < \text{tol}_G
            b. Monotonicity: k_0 = \arg\min \hat{G}(k) Stop if \hat{G}(k) > \hat{G}(k-1) > \hat{G}(k-2)
        4. Sets the convergence flag and updates alpha = alpha_{k_0} if criteria are met.
        ----------------------------------------------------------------------------------
        """
        super()._run_convergence_checks()

        if len(self.Ghat_history) > 3:
            if (
                abs(self.Ghat_history[-1] - self.Ghat_history[-2])
                / abs(self.Ghat_history[0])
                < self.tol
            ):
                log.debug(
                    "GHat: The regularisation parameter has converged at outer iteration",
                    self.iteration,
                )
                self.converged = True
            elif (
                self.Ghat_history[-1] > self.Ghat_history[-2]
                and self.Ghat_history[-2] > self.Ghat_history[-3]
            ):
                self.iteration = np.argmin(self.Ghat_history)
                self.regalpha = self.regalpha_history[self.iteration]

                log.debug(
                    "The regularisation parameter has converged at outer iteration",
                    self.iteration,
                )
                self.converged = True

    def plot_function(
        self,
        regalpha_limits: Optional[Tuple[float, float]] = None,
        num_points: int = 80,
        filepath: Optional[str] = None,
    ):
        r"""
        Plot the GCV function over a range of regularization parameters.

        Args:
            regalpha_limits (Optional[Tuple[float, float]]):
                The (min, max) range for :math:`\alpha`. Defaults to (self.regalpha_low, self.regalpha_high).
            num_points (int):
                Number of points in the grid. Defaults to 80.
            filepath (Optional[str]):
                If provided, saves the plot to the specified file path.
        """
        # Generate grid data and evaluate GCV function
        regalpha_grid, func_grid, reg_idx = self._geometric_grid(
            regalpha_limits, num_points
        )

        # Find the peak from the grid, avoiding the very first/last points
        # to bypass the boundary plateaus seen in your plots.
        grid_search_idx = np.argmin(func_grid)
        regalpha_grid[grid_search_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(
            regalpha_grid, func_grid, label=f"{self.gcv_type.upper()} GCV Function"
        )
        ax.semilogx(
            regalpha_grid[reg_idx],
            func_grid[reg_idx],
            "ro",
            markersize=8,
            label=rf"$\alpha={self.regalpha:.3e}$",
        )
        ax.semilogx(
            regalpha_grid[grid_search_idx],
            func_grid[grid_search_idx],
            "bo",
            markersize=8,
            label=rf"Grid Search $\alpha={regalpha_grid[grid_search_idx]:.3e}$",
        )
        ax.axvline(self.regalpha, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Regularisation parameter ($\\alpha$)")
        ax.set_ylabel("GCV Function Value")
        ax.set_title(
            f"{self.gcv_type.upper()} GCV Function and Weight ($\omega={self.omega:.3e}$)"
        )
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        if filepath:
            plt.savefig(filepath)
        plt.show()
        plt.close(fig)

class grid_search_mixin():
    def grid_search(self,regalpha_limits: Optional[Tuple[float, float]] = None, 
                    num_points: int = 200):
        r"""
        Perform a grid search over a range of regularization.

        Args:
            num_points (int):
                Number of points in the grid. Defaults to 80.

        Returns:
            Tuple[float, float, np.ndarray, np.ndarray]:
                - alpha_grid_search: The regularization parameter that maximizes curvature.
                - func_grid_search: The maximum curvature value.
                - func_grid: Array of function values over the grid.
                - regalpha_grid: Array of regularization parameters over the grid.

        """
        regalpha_grid, func_grid, _ = self._geometric_grid(regalpha_limits=regalpha_limits, 
                                                           num_points=num_points)

        sign_change_idxs = []

        dphi = np.array([self._func_first_derivative(a) for a in regalpha_grid])

        # Remove invalid entries
        valid = np.isfinite(dphi)
        regalpha_grid, dphi = regalpha_grid[valid], dphi[valid]


        # Find sign changes
        sign_change_idxs = np.where(np.sign(dphi[:-1]) != np.sign(dphi[1:]))[0]

        if len(sign_change_idxs) == 0:
            # No sign changes: use the grid peak
            alpha_grid_search = regalpha_grid[np.argmin(func_grid)]
            func_grid_search = self.func(alpha_grid_search)
        else:
            # Idx for sign changes
            func_sign_changes = func_grid[sign_change_idxs]
            regalpha_sign_change = regalpha_grid[sign_change_idxs]
            alpha_grid_search = regalpha_sign_change[np.argmin(func_sign_changes)]
            func_grid_search = self.func(alpha_grid_search)
        
        return alpha_grid_search, func_grid_search, func_grid, regalpha_grid


class UpdateRegLcurve(UpdateRegBase, grid_search_mixin):
    r"""L-curve method for choosing the regularisation parameter :math:`\alpha`.

    This rule identifies the "corner" of the L-curve—the point of maximum curvature
    when plotting the log-norm of the solution versus the log-norm of the residual.

    In the :math:`k`-th Krylov subspace, the performance of the regularised
    solution :math:`x_\alpha` is evaluated using the following projected norms:

    .. math:: \eta(\alpha) = \log \|x_\alpha\|_2, \quad \rho(\alpha) = \log \|r_\alpha\|_2

    These are the logarithms of the projected solution and residual norms, respectively.
    See :meth:`_projected_solution_norm_sq` and :meth:`_projected_residual_norm_sq` for details.

    The curvature :math:`\kappa(\alpha)` is computed using exact analytical first
    (:math:`\rho', \eta'`) and second (:math:`\rho'', \eta''`) derivatives of
    these log-norms with respect to :math:`\alpha`:

    .. math:: \kappa(\alpha) = \frac{\rho' \eta'' - \eta' \rho''}{((\rho')^2 + (\eta')^2)^{3/2}}

    The algorithm finds :math:`\alpha` that maximizes the curvature :math:`\kappa(\alpha)` using
    a bounded optimization search within the range of the current projected
    singular values.

    Parameters
    ----------
    tol : float
        Relative tolerance for convergence of the regularisation parameter.
    data_size : int
        Number of elements in the data; corresponds to range of the operator
        (denoted as 'm').
    domain_size : int
        Number of elements in the solution; corresponds to domain of the operator
        (denoted as 'n').
    """

    def __init__(self, data_size: int, domain_size: int, tol: float = 1e-2):
        super().__init__(data_size, domain_size, tol)
        self.rule_type = "l-curve"

    def _compute_next_regalpha(self):
        """
        Find alpha that maximizes curvature using a global nonconvex 1D optimizer.
        Handles negative or nonconvex functions efficiently for scalar reg.
        """
        # Grid search for a robust starting point
        x0, _, _, _ = self.grid_search()

        # 3. Local Refinement: Bounded Optimization
        res = scipy.optimize.minimize(
            lambda reg: self.func(reg),
            x0=x0,
            bounds=[(self.regalpha_low, self.regalpha_high)],
            tol=1e-10,
        )

        if res.success and np.isfinite(res.x[0]):
            new_alpha = res.x[0]
        else:
            # Fallback to the grid search peak if the optimizer fails to converge
            new_alpha = x0

        return new_alpha

    def func(self, reg):
        r"""Compute the L-curve curvature :math:`\kappa(\alpha)` at a specific alpha.

        This method implements the log-log space curvature formula. Although the
        L-curve is defined in :math:`(\log\|r\|, \log\|x\|)` space, this function
        computes the required derivatives analytically. This approach is more
        numerically stable than computing finite-difference derivatives of the
        log-norms, as the logarithmic functions are eliminated through differentiation,
        avoiding potential issues with small values.

        Given :math:`\rho = \log \|r\| = \frac{1}{2} \log(R^2)`, the derivatives
        w.r.t. :math:`\alpha` are:

        .. math::
           \rho' = \frac{(R^2)'}{2 R^2}, \quad
           \rho'' = \frac{R^2 (R^2)'' - ((R^2)')^2}{2 (R^2)^2}

        The same logic applies to the solution norm :math:`\eta` for :math:`\eta`
        and :math:`\rho` defined in classes :class:`UpdateRegLcurve` docs.

        Parameters
        ----------
        reg : float
            The regularization parameter :math:`\alpha`.

        Returns
        -------
        float
            The curvature :math:`\kappa` at the given :math:`\alpha`.
        """

        # Squared Norms of Squared Projected Norms
        R2 = self._projected_residual_norm_sq(reg)
        X2 = self._projected_solution_norm_sq(reg)

        # Avoid divide-by-zero
        if R2 <= self.EPS or X2 <= self.EPS:
            return 0.0

        # First and Second Derivatives of Squared Projected Norms
        (R2_p, X2_p) = self._projected_norm_first_derivatives(reg)
        (R2_pp, X2_pp) = self._projected_norm_second_derivatives(reg)

        # Log-derivatives (Chain rule applied to log(sqrt(NormSq)))
        logR_p = 0.5 * R2_p / R2
        logX_p = 0.5 * X2_p / X2

        logR_pp = (R2 * R2_pp - R2_p**2) / (2 * R2**2)
        logX_pp = (X2 * X2_pp - X2_p**2) / (2 * X2**2)

        # Curvature formula for a parametric curve
        num = logR_p * logX_pp - logX_p * logR_pp
        denom = (logR_p**2 + logX_p**2) ** 1.5 + 1e-300

        return - num / denom
    
    def _func_first_derivative(self, reg):
        R2 = self._projected_residual_norm_sq(reg)
        X2 = self._projected_solution_norm_sq(reg)
        
        (R2_p, X2_p) = self._projected_norm_first_derivatives(reg)
        (R2_pp, X2_pp) = self._projected_norm_second_derivatives(reg)
        (R2_ppp, X2_ppp) = self._projected_norm_third_derivatives(reg)
        
        rho_p = 0.5 * R2_p / R2
        rho_pp = (R2 * R2_pp - R2_p**2) / (2 * R2**2)
        rho_ppp = (R2 * R2_ppp - 3 * R2_p * R2_pp + 2 * R2_p**3 / R2) / (2 * R2**2)
        
        eta_p = 0.5 * X2_p / X2
        eta_pp = (X2 * X2_pp - X2_p**2) / (2 * X2**2)
        eta_ppp = (X2 * X2_ppp - 3 * X2_p * X2_pp + 2 * X2_p**3 / X2) / (2 * X2**2)
        
        num = (rho_p * eta_ppp - eta_p * rho_ppp) * (rho_p**2 + eta_p**2)
        denom = (rho_p**2 + eta_p**2)**(5/2)
        correction = 3 * (rho_p * eta_pp - eta_p * rho_pp) * (rho_p * rho_pp + eta_p * eta_pp)
        
        return (num - correction) / denom

    def plot_function(
        self,
        regalpha_limits: Optional[Tuple[float, float]] = None,
        num_points: int = 80,
        filepath: Optional[str] = None,
    ):
        r"""
        Plot the L-curve function over a range of regularization parameters.

        Args:
            regalpha_limits (Optional[Tuple[float, float]]):
                The (min, max) range for :math:`\alpha`. Defaults to (self.regalpha_low, self.regalpha_high).
            num_points (int):
                Number of points in the grid. Defaults to 80.
            filepath (Optional[str]):


        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - regalpha_grid: Array of shape (num_points,) containing :math:`\alpha` values.
                - func_grid: Array of shape (num_points,) containing function values.

        Raises:
            ValueError: If the limits are negative or if the upper bound is not
                        strictly greater than the lower bound.
        """
        # Use provided axis or create one
        fig = plt.figure(figsize=(12, 5))

        # Grid search for a robust starting point
        alpha_grid_search, func_grid_search, func_grid, regalpha_grid = self.grid_search(
            regalpha_limits=regalpha_limits,
            num_points=num_points
        )

        # Compute Plot Grid Values
        r_vals_grid = [self._projected_residual_norm_sq(reg) for reg in regalpha_grid]
        x_vals_grid = [self._projected_solution_norm_sq(reg) for reg in regalpha_grid]

        # Plot the L-curve
        ax = fig.add_subplot(1, 2, 1)
        ax.loglog(r_vals_grid, x_vals_grid, linestyle="-")

        # Highlight the chosen alpha
        ax.loglog(
            self._projected_residual_norm_sq(self.regalpha),
            self._projected_solution_norm_sq(self.regalpha),
            "ro",
            markersize=8,
            label=rf"$\alpha={self.regalpha:.3e}$",
        )
        ax.loglog(
            self._projected_residual_norm_sq(alpha_grid_search),
            self._projected_solution_norm_sq(alpha_grid_search),
            "bo",
            markersize=8,
            label=rf"Grid Max $\alpha={alpha_grid_search:.3e}$",
        )
        ax.set_xlabel(r"$\|B_k y(\alpha)-\|b\|_2 e_1\|_2$")
        ax.set_ylabel(r"$\|x(\alpha)\|_2$")
        ax.set_title("L-curve (projected)")
        ax.legend()

        # Plot the curvature function
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.semilogx(regalpha_grid, -func_grid, linestyle="-")
        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel("Curvature")
        ax2.set_title("L-curve curvature (projected)")
        ax2.set_xlim(regalpha_grid[0], regalpha_grid[-1])
        # Highlight chosen alpha point
        ax2.semilogx(
            self.regalpha,
            -self.func(self.regalpha),
            "ro",
            markersize=8,
            label=rf"$\alpha={self.regalpha:.3e}$",
        )
        ax2.semilogx(
            alpha_grid_search,
            -func_grid_search,
            "bo",
            markersize=8,
            label=rf"Grid Max $\alpha={alpha_grid_search:.3e}$",
        )
        ax2.legend()
        # Dotted vertical line at chosen alpha
        ax2.axvline(self.regalpha, color="gray", linestyle="--")
        plt.show()
        if filepath:
            plt.savefig(filepath)
        plt.close(fig)

class UpdateRegReginska(UpdateRegBase, grid_search_mixin):
    r"""Reginska's Rule for choosing the regularization parameter :math:`\alpha`.

    This rule identifies the optimal :math:`\alpha` by minimizing the functional:
    .. math:: \Phi(\alpha) = \log(\|r_\alpha\|_2^2) + \mu \log(\|x_\alpha\|_2^2)

    where :math:`r_\alpha` is the projected residual, :math:`x_\alpha` is the
    projected solution, and :math:`\mu` is a user-defined parameter that balances
    the influence of the solution norm.

    Parameters
    ----------
    tol : float
        Relative tolerance for convergence of the regularisation parameter.
    data_size : int
        Number of elements in the data; corresponds to range of the operator
        (denoted as 'm').
    domain_size : int
        Number of elements in the solution; corresponds to domain of the operator
        (denoted as 'n').
    mu : float, optional
        The balancing parameter :math:`\mu` in the Reginska functional. Defaults to 1.0. Increasing
        :math:`\mu` places more emphasis on minimizing the solution norm.
    """

    def __init__(self, data_size: int, domain_size: int, tol: float = 1e-2, mu: float = 1.0):
        super().__init__(data_size, domain_size, tol)
        self.rule_type = "reginska"
        self.mu = mu

    def _compute_next_regalpha(self):
        """
        Compute the next regularization parameter using derivative-based
        Regińska minimization.

        Strategy:
        1. Sample Phi'(alpha) on a log grid.
        2. Detect sign changes (candidate stationary points).
        3. Refine roots with Brent's method.
        4. Reject flat / unstable minima.
        """
        # Grid search for a starting point
        x0, _, _, _ = self.grid_search()

        # Local Refinement: Bounded Optimization
        res = scipy.optimize.minimize(
            self.func,
            x0=x0,
            bounds=[(self.regalpha_low, self.regalpha_high)],
            tol=1e-10
        )
        return res.x[0] if res.success and np.isfinite(res.x[0]) else x0


    def func(self, regalpha):
        """Log-Reginska functional: log(R^2) + mu * log(X^2)."""
        r2 = self._projected_residual_norm_sq(regalpha)
        x2 = self._projected_solution_norm_sq(regalpha)
        
        if r2 <= self.EPS or x2 <= self.EPS:
            return 1e30 
            
        return 0.5 * (np.log(r2) + self.mu * np.log(x2))

    def _func_first_derivative(self, regalpha):
        """
        Derivative of the Log-Reginska functional with respect to alpha.

        .. math::
            `\Phi(\alpha) = \tfrac12(\log\|r_\alpha\|_2^2 + \mu \log\|x_\alpha\|_2^2),
            \quad
            \frac{d\Phi}{d\alpha}
            = \tfrac12\left(
            \frac{1}{\|r_\alpha\|_2^2}\frac{d}{d\alpha}\|r_\alpha\|_2^2
            + \mu \frac{1}{\|x_\alpha\|_2^2}\frac{d}{d\alpha}\|x_\alpha\|_2^2
            \right)`
        
        Returns
        -------
        float
            The derivative value.
        """
        R2 = self._projected_residual_norm_sq(regalpha)
        X2 = self._projected_solution_norm_sq(regalpha)
        (R2_p, X2_p) = self._projected_norm_first_derivatives(regalpha)

        # Use a small epsilon for the denominator instead of a hard jump to 1e30
        denom_r = max(R2, self.EPS)
        denom_x = max(X2, self.EPS)

        return 0.5 * (R2_p / denom_r + self.mu * X2_p / denom_x)

    def plot_function(self, regalpha_limits=None, num_points=80, filepath=None):
        """
        Plot the Reginska functional and L-curve over a range of regularization parameters.
        """
        # Grid search for a robust starting point
        alpha_grid_search, func_grid_search, func_grid, regalpha_grid = self.grid_search(
            regalpha_limits=regalpha_limits,
            num_points=num_points
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Compute Plot Grid Values
        r_vals_grid = [self._projected_residual_norm_sq(reg) for reg in regalpha_grid]
        x_vals_grid = [self._projected_solution_norm_sq(reg) for reg in regalpha_grid]

        # Plot the L-curve
        ax1.loglog(r_vals_grid, x_vals_grid, linestyle="-")

        # Highlight the chosen alpha
        ax1.loglog(
            self._projected_residual_norm_sq(self.regalpha),
            self._projected_solution_norm_sq(self.regalpha),
            "ro",
            markersize=8,
            label=rf"$\alpha={self.regalpha:.3e}$",
        )
        ax1.loglog(
            self._projected_residual_norm_sq(alpha_grid_search),
            self._projected_solution_norm_sq(alpha_grid_search),
            "bo",
            markersize=8,
            label=rf"Grid Max $\alpha={alpha_grid_search:.3e}$",
        )
        ax1.set_xlabel(r"$\|B_k y(\alpha)-\|b\|_2 e_1\|_2$")
        ax1.set_ylabel(r"$\|x(\alpha)\|_2$")
        ax1.set_title("L-curve (projected)")
        ax1.legend()
        
        
        # Plot exp(func) to show the original Psi scale
        ax2.loglog(regalpha_grid, np.exp(func_grid), color='tab:red', label=r"$\Psi(\alpha)$")
        ax2.set_title(rf"Reginska Functional ($\mu={self.mu}$)")
        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(r"$\Psi(\alpha)$")
        ax2.semilogx(
            self.regalpha,
            np.exp(self.func(self.regalpha)),
            "ro",
            markersize=8,
            label=rf"$\alpha={self.regalpha:.3e}$",
        )
        ax2.semilogx(
            alpha_grid_search,
            np.exp(func_grid_search),
            "bo",
            markersize=8,
            label=rf"Grid Max $\alpha={alpha_grid_search:.3e}$",
        )
        ax2.legend()
        ax2.axvline(self.regalpha, color="gray", linestyle=":")

        plt.tight_layout()
        if filepath: plt.savefig(filepath)
        plt.show()
        plt.close(fig)

class UpdateRegUPRE(UpdateRegBase):
    r"""Unbiased Predictive Risk Estimator (UPRE) for choosing :math:`\alpha`.

    .. math:: U(\alpha) = \frac{1}{m} \|r_\alpha\|_2^2 + \frac{2\sigma^2}{m} \text{trace}(A_\alpha) - \sigma^2
    """

    def __init__(self, data_size: int, domain_size: int, tol: float = 1e-2, noise_variance: float = 0.0):
        super().__init__(data_size, domain_size, tol)
        self.rule_type = "upre"
        self.sigma2 = noise_variance

    def _compute_next_regalpha(self):
        regalpha_grid, func_grid, _ = self._geometric_grid()
        x0 = regalpha_grid[np.argmin(func_grid)]

        res = scipy.optimize.minimize(
            self.func,
            x0=x0,
            bounds=[(self.regalpha_low + self.EPS, self.regalpha_high)],
            tol=1e-10
        )
        return res.x[0] if res.success else x0

    def func(self, regalpha):
        r2 = self._projected_residual_norm_sq(regalpha)
        trace_A = np.sum(self.Sbsq / (self.Sbsq + regalpha**2))
        return (1.0 / self.m) * r2 + (2.0 * self.sigma2 / self.m) * trace_A - self.sigma2

    def plot_function(self, regalpha_limits=None, num_points=80, filepath=None):
        regalpha_grid, func_grid, reg_idx = self._geometric_grid(regalpha_limits, num_points)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(regalpha_grid, func_grid, label="UPRE $U(\\alpha)$")
        ax.semilogx(regalpha_grid[reg_idx], func_grid[reg_idx], "ro", label=rf"$\alpha={self.regalpha:.3e}$")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$U(\alpha)$")
        ax.set_title("UPRE Function Minimization")
        ax.grid(True, which="both", alpha=0.5)
        ax.legend()
        if filepath:
            plt.savefig(filepath)
        plt.show()