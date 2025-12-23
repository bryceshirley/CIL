import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class UpdateRegBase(ABC):
    '''Base class for updating regularization parameters in iterative solvers.

    Parameters
    ----------
    regalpha_saturation_tol : float
        Relative tolerance for convergence.
    m : int
        Number of rows in the forward operator.
    n : int
        Number of columns in the forward operator.
    '''

    def __init__(self, regalpha_saturation_tol, m, n):
        # Validation for initialization parameters
        if m <= 0 or n <= 0:
            raise ValueError(f"Dimensions m and n must be positive integers. Got m={m}, n={n}")
        if regalpha_saturation_tol <= 0:
            raise ValueError(f"Tolerance must be greater than 0. Got {regalpha_saturation_tol}")

        self.rule_type = "base rule"
        self.m = m
        self.n = n
        self.tol = regalpha_saturation_tol
        
        # History and state variables
        self.regalpha = 0.0
        self.regalpha_history = [] # History of regularization parameters
        self.func_history = [] # History of function values

        self.converged = False
        self.iteration = 0
        self.EPS = 1e-12 

        # Bounds Based on SVD of Bk
        self.regalpha_low = 0.0
        self.regalpha_high = None

    def update_regularizationparam(self, Bk, b_norm):
        '''Main entry point to update the regularization parameter.
        
        Parameters
        ----------
        Bk : np.ndarray
            The projection matrix (k+1 x k).
        b_norm : float
            The norm of the initial right-hand side (data) vector.
        '''
        # Validation for runtime inputs
        if not isinstance(Bk, np.ndarray):
            raise TypeError(f"Bk must be a numpy.ndarray, got {type(Bk)}")
            
        k_plus_1, k = Bk.shape
        if k_plus_1 != k + 1:
            raise ValueError(f"Bk must have shape (k+1, k). Got {Bk.shape}")
        
        if k > self.n or k_plus_1 > self.m:
            raise ValueError(f"Krylov subspace size {k} cannot exceed operator dimensions (m={self.m}, n={self.n})")

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
            if np.isscalar(func_value) and func_value is not None and not np.isnan(func_value) and not np.isinf(func_value):
                self.func_history.append(func_value)
            
            self._run_convergence_checks()
        self._print_status()
    
    def _residual_filter(self, reg):
        '''
        Computes the residual filter factors for a given regularization parameter.

        The residual filter, often denoted as :math:`f_r(s,\alpha)`, dampens the 
        influence of small singular values on the residual. For Tikhonov 
        regularization, it is defined as:

        .. math::

            f_r(s,\alpha) = \frac{\alpha^2}{s^2 + \alpha^2}

        Where:
        * :math:`\alpha` is the regularization parameter (``reg``).
        * :math:`s` are the singular values of the system (``Sb``).
        '''
        return (reg**2) / (self.Sbsq + reg**2)
    
    def _solution_filter(self, reg):
        '''
        Computes the solution filter factors for a given regularization parameter.

        The solution filter, or filter function :math:`f_x`, scales the 
        singular values to provide a smoothed pseudo-inverse solution:

        .. math::

            f_x(s,\alpha) = \frac{s}{s^2 + \alpha^2}

        This filter ensures that as :math:`s \to 0`, the solution does not 
        blow up, effectively suppressing noise in the inversion process.
        '''
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

    def _run_convergence_checks(self):
        '''Standard convergence check based on the saturation of alpha.
        
        Uses a relative change formula with a safety epsilon EPS.
        '''
        if len(self.regalpha_history) < 2:
            return
        denom = abs(self.regalpha_history[-1]) + self.EPS
        rel_change = abs(self.regalpha_history[-1] - self.regalpha_history[-2]) / denom
        
        if rel_change < self.tol:
            print(f"Alpha Saturation: Converged at iteration {self.iteration}")
            self.converged = True

    @abstractmethod
    def _compute_next_regalpha(self):
        '''Abstract method to compute the next regularization parameter.

        Returns
        -------
        new_regalpha : float
            The updated regularization parameter.
        '''
        pass

    @abstractmethod
    def func(self, regalpha):
        '''Abstract method to compute the rule-specific function value.

        Parameters
        ----------
        regalpha : float
            The regularization parameter at which to evaluate the function.

        Returns
        -------
        value : float
            The computed function value.
        '''
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
        '''
        Utility to plot the history of regularization parameters.

        Parameters
        ----------
        show_objective : bool, optional
            If True, plots both the alpha history and the objective function 
            value history in subplots. Default is False.
        '''
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
        ax_alpha.plot(self.regalpha_history, marker='o', color='tab:blue', label=r'$\alpha$')
        ax_alpha.set_ylabel(r'Regularization $\alpha$')
        ax_alpha.set_yscale('log')
        ax_alpha.set_title(f'{self.rule_type.upper()} Regularization Parameter History')
        ax_alpha.grid(True, which="both", ls="-", alpha=0.5)

        if show_objective:
            # Subplot 2: Objective History
            ax_func = axes[1]
            iters = np.arange(len(self.func_history))
            ax_func.plot(iters, self.func_history, marker='x', color='tab:red', linestyle='--')
            ax_func.set_ylabel(f'{self.rule_type.upper()} Function History')
            ax_func.set_xlabel('Iteration')
            ax_func.grid(True, alpha=0.5)
        else:
            ax_alpha.set_xlabel('Iteration')

        plt.tight_layout()
        plt.show()
    
    def _geometric_grid(self, regalpha_limits: Optional[Tuple[float, float]] = None, num_points: int = 80):
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
            high = max(self.regalpha_high, low * 10.0) # Ensure a valid range
            regalpha_limits = (low, high)

        start, stop = regalpha_limits

        # Geometric space requires strictly positive bounds
        if start <= 0:
            start = self.EPS
        
        if stop <= start:
            raise ValueError(f"Upper bound ({stop}) must be greater than lower bound ({start}).")

        # 1. Generate the grid
        regalpha_grid = np.geomspace(start, stop, num_points)

        # 2. Evaluate the function 
        # Using np.array for the result for easier downstream slicing/plotting
        func_grid = np.array([self.func(reg) for reg in regalpha_grid])

        # Find index of chosen alpha in grid
        reg_idx = int(np.argmin(np.abs(regalpha_grid - self.regalpha)))

        return regalpha_grid, func_grid, reg_idx

    def plot_function(self, regalpha_limits: Optional[Tuple[float, float]] = None):
        '''Utility to plot the rule-specific function over a range of alphas.
        Parameters
        ----------
        regalpha_limits : Optional[Tuple[float, float]]
            Tuple specifying the (min, max) range of regularization parameters to plot. 
            If None, defaults to the class's regalpha_low and regalpha_high.
        '''
        print("Plotting function not implemented")

    def _print_status(self):
        '''Prints the current iteration status to the console.'''
        val = f"{self.regalpha:.4e}" if self.regalpha else "N/A"
        print(f"Iteration {self.iteration}: regalpha = {val} [{self.rule_type.upper()}]")
    

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
    regalpha_saturation_tol : float
        Relative tolerance for convergence of the regularisation parameter.
    m : int
        Number of rows in the forward operator :math:`A`.
    n : int
        Number of columns in the forward operator :math:`A`.
    noise_level_estimate : float
        The estimated norm of the noise in the right-hand side, :math:`\eta = \|e\|_2`.

    Note
    ----
    The discrepancy principle requires an accurate estimate of the noise level. 
    If :math:`\eta` is underestimated, the solution may be under-regularised; 
    if overestimated, the solution will be over-smoothed.

    """

    def __init__(self, regalpha_saturation_tol, m, n, noise_level_estimate=0.0):
        super().__init__(regalpha_saturation_tol, m, n)
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
            return self.regalpha_low # Return 0.0

        # If f_hi < 0, it means even with max smoothing, 
        # the residual is smaller than the noise level.
        if f_hi < 0:
            return self.regalpha_high

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
    
    def plot_function(self, regalpha_limits: Optional[Tuple[float, float]] = None, 
                      num_points: int = 80,
                      filepath: Optional[str] = None):
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
        regalpha_grid, func_grid, reg_idx = self._geometric_grid(regalpha_limits, num_points)
        
        # Calculate residual norms for the second plot
        # ||r|| = sqrt(phi(alpha) + eta^2)
        res_norm_grid = np.sqrt(func_grid + self.noise_level_estimate**2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # --- Subplot 1: Discrepancy Function ---
        ax1.semilogx(regalpha_grid, func_grid, label='Discrepancy $\phi(\\alpha)$')
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax1.semilogx(regalpha_grid[reg_idx], func_grid[reg_idx], "ro", 
                     markersize=8, label=rf"$\alpha={self.regalpha:.3e}$")
        ax1.axvline(self.regalpha, color="gray", linestyle=":", alpha=0.5)
        ax1.set_xlabel("Regularisation parameter ($\\alpha$)")
        ax1.set_ylabel("$\|r_\\alpha\|_2^2 - \eta^2$")
        ax1.set_title("Discrepancy Function Root")
        ax1.legend()
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

        # --- Subplot 2: Residual Norm vs Noise Level ---
        ax2.semilogx(regalpha_grid, res_norm_grid, label="Residual Norm $\|r_\\alpha\|_2$")
        # Horizontal line for noise level estimate
        ax2.axhline(self.noise_level_estimate, color="green", linestyle="--", 
                    label=rf"Noise Level $\eta={self.noise_level_estimate:.2e}$")
        ax2.semilogx(regalpha_grid[reg_idx], res_norm_grid[reg_idx], "ro", markersize=8)
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

class UpdateRegLcurve(UpdateRegBase):
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
    regalpha_saturation_tol : float
        Relative tolerance for convergence of the regularisation parameter.
    m : int
        Number of rows in the forward operator :math:`A`.
    n : int
        Number of columns in the forward operator :math:`A`.
    """
    def __init__(self, regalpha_saturation_tol, m, n):
        super().__init__(regalpha_saturation_tol, m, n)
        self.rule_type = "l-curve"

    def _compute_next_regalpha(self):
        """
        Find alpha that maximizes curvature using a global nonconvex 1D optimizer.
        Handles negative or nonconvex functions efficiently for scalar reg.
        """
        self.regalpha_high = self.Sbmax
        self.regalpha_low = max(self.Sbmin, self.EPS)
        # 1. Setup Search Bounds in log10 space
        # Using Sbmin/Sbmax ensures we stay within the meaningful spectral range.
        log_low = np.log10(self.regalpha_low)
        log_high = np.log10(self.regalpha_high)

        # 2. Global Search: Coarse Grid
        # Used to distinguish the elbow from boundary artifacts.
        grid_alphas = np.logspace(log_low, log_high, num=50)
        grid_curvatures = np.array([self.func(a) for a in grid_alphas])
        
        # Find the peak from the grid, avoiding the very first/last points 
        # to bypass the boundary plateaus seen in your plots.
        inner_idx = np.argmax(grid_curvatures[5:-5]) + 5
        x0_refined = grid_alphas[inner_idx]

        # 3. Local Refinement: Bounded Optimization
        # We maximize the absolute curvature to handle potential sign flips in noise.
        res = scipy.optimize.minimize(
            lambda reg: -abs(self.func(reg)),
            x0=x0_refined, 
            bounds=[(10**log_low, 10**log_high)],
            tol=1e-10,
            options={'ftol': 1e-14}
        )

        if res.success and np.isfinite(res.x[0]):
            new_alpha = res.x[0]
        else:
            # Fallback to the grid search peak if the optimizer fails to converge
            new_alpha = x0_refined

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
        (R2_p, R2_pp, X2_p, X2_pp) = self._projected_norm_derivatives(reg)

        # Log-derivatives (Chain rule applied to log(sqrt(NormSq)))
        logR_p = 0.5 * R2_p / R2
        logX_p = 0.5 * X2_p / X2

        logR_pp = (R2 * R2_pp - R2_p**2) / (2 * R2 **2)
        logX_pp = (X2 * X2_pp - X2_p**2) / (2 * X2 **2)

        # Curvature formula for a parametric curve
        num = logR_p * logX_pp - logX_p * logR_pp
        denom = (logR_p**2 + logX_p**2) ** 1.5 + 1e-300
        
        return num / denom

    def _projected_norm_derivatives(self, reg):
        r"""Compute analytical squared projected norms and their 1st/2nd derivatives.

        **Variable Naming Convention:**
        
        In the code, suffixes are used to represent derivatives with respect 
        to the regularization parameter :math:`\alpha`:
        
        * **_p**: First derivative, :math:`\frac{d}{d\alpha}` (e.g., ``R2_p`` is :math:`(R^2)'`).
        * **_pp**: Second derivative, :math:`\frac{d^2}{d\alpha^2}` (e.g., ``R2_pp`` is :math:`(R^2)''`).

        Returns
        -------
        tuple
            (R2_p, R2_pp, X2_p, X2_pp)
        """
        # SVD-based filter factors
        fr, fx, fr_p, fr_pp, fx_p, fx_pp = self._filter_derivatives(reg)

        # First and Second Derivatives of Squared Projected Norms
        b_norm_sq = self.b_norm * self.b_norm
        u1_sq = self.u1 * self.u1

        R2_p = 2 * b_norm_sq * np.sum(u1_sq * fr * fr_p)
        X2_p = 2 * b_norm_sq * np.sum(u1_sq * fx * fx_p)

        R2_pp = 2 * b_norm_sq * np.sum(u1_sq * (fr_p**2 + fr * fr_pp))
        X2_pp = 2 * b_norm_sq * np.sum(u1_sq * (fx_p**2 + fx * fx_pp))
        return R2_p, R2_pp, X2_p, X2_pp

    def _filter_derivatives(self, reg):
        r"""
        Compute first and second derivatives of Tikhonov filter factors.

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
        * **_pp**: Second derivative, :math:`\frac{d^2}{d\alpha^2}` (e.g., ``R2_pp`` is :math:`(R^2)''`).


        Returns
        -------
        tuple
            (fr, fx, fr_p, fr_pp, fx_p, fx_pp)
        """
        # Precompute reusable squared terms
        reg2 = reg**2

        # SVD-based filter factors
        fr = self._residual_filter(reg)
        fx = self._solution_filter(reg)

        # Common denominator for Filters 
        f_denom = (self.Sbsq + reg2)

        # First and Second Derivatives of Filter w.r.t alpha 
        fr_p = 2 * reg * self.Sbsq / f_denom**2
        fx_p = -2 * reg * self.Sb / f_denom**2
        fr_pp = 2 * self.Sbsq * (self.Sbsq - 3 * reg2) / f_denom**3
        fx_pp = 2 * self.Sb * (3 * reg2 - self.Sbsq) / f_denom**3
        return fr, fx, fr_p, fr_pp, fx_p, fx_pp

    def plot_function(self, regalpha_limits: Optional[Tuple[float, float]] = None, 
                      num_points: int = 80,
                      filepath: Optional[str] = None):
        r"""
        Plot the discrepancy function over a range of regularization parameters.

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
        regalpha_grid, func_grid, reg_idx = self._geometric_grid(regalpha_limits, num_points)

        # Use provided axis or create one
        fig = plt.figure(figsize=(12, 5))

        # Grid Search max
        grid_max_idx = np.argmax(func_grid)

        # Compute Plot Grid Values
        r_vals_grid = [self._projected_residual_norm_sq(reg) for reg in regalpha_grid]
        x_vals_grid = [self._projected_solution_norm_sq(reg) for reg in regalpha_grid]

        # Plot the L-curve
        ax = fig.add_subplot(1, 2, 1)
        ax.loglog(r_vals_grid, x_vals_grid, linestyle="-")

        # Highlight the chosen alpha
        ax.loglog(r_vals_grid[reg_idx], x_vals_grid[reg_idx], "ro",
            markersize=8,
            label=rf"$\alpha={self.regalpha:.3e}$",
        )
        ax.loglog(r_vals_grid[grid_max_idx], x_vals_grid[grid_max_idx], "bo",
            markersize=8,
            label=rf"Grid Max $\alpha={regalpha_grid[grid_max_idx]:.3e}$",
        )
        ax.set_xlabel(r"$\|B_k y(\alpha)-\|b\|_2 e_1\|_2$")
        ax.set_ylabel(r"$\|x(\alpha)\|_2$")
        ax.set_title("L-curve (projected)")
        ax.legend()

        # Plot the curvature function
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.semilogx(
            regalpha_grid, func_grid,
            linestyle="-"
        )
        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel("Curvature")
        ax2.set_title("L-curve curvature (projected)")
        ax2.set_xlim(regalpha_grid[0], regalpha_grid[-1])
        # Highlight chosen alpha point
        ax2.semilogx(
            regalpha_grid[reg_idx],
            func_grid[reg_idx],
            "ro",
            markersize=8,
            label=rf"$\alpha={self.regalpha:.3e}$",
        )
        ax2.semilogx(
            regalpha_grid[grid_max_idx],
            func_grid[grid_max_idx],
            "bo",
            markersize=8,
            label=rf"Grid Max $\alpha={regalpha_grid[grid_max_idx]:.3e}$",
        )
        ax2.legend()
        # Dotted vertical line at chosen alpha
        ax2.axvline(self.regalpha, color="gray", linestyle="--")
        plt.show()
        if filepath:
            plt.savefig(filepath)
        plt.close(fig)

class UpdateRegGCV(UpdateRegBase):
    def __init__(self, regalpha_saturation_tol, m, n, gcv_type='weighted'):
        super().__init__(regalpha_saturation_tol, m, n)
        if gcv_type not in ['weighted','adaptive-weighted', 'standard']:
            raise ValueError(f"gcv_type must be one of 'weighted', 'adaptive-weighted', or 'standard'. Got {gcv_type}")
        self.rule_type = f"{gcv_type} gcv"
    def _compute_next_regalpha(self):
        return 0.0
    def func(self, regalpha):
        return 0.0
