import numpy as np
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

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
        self.EPS = 1e-15 

        # Bounds Based on SVD of Bk
        self.regalpha_low = None
        self.regalpha_high = None

    def update_regularizationparam(self, Bk, beta0):
        '''Main entry point to update the regularization parameter.
        
        Parameters
        ----------
        Bk : np.ndarray
            The projection matrix (k+1 x k).
        beta0 : float
            The initial residual norm.
        '''
        # Validation for runtime inputs
        if not isinstance(Bk, np.ndarray):
            raise TypeError(f"Bk must be a numpy.ndarray, got {type(Bk)}")
            
        k_plus_1, k = Bk.shape
        if k_plus_1 != k + 1:
            raise ValueError(f"Bk must have shape (k+1, k). Got {Bk.shape}")
        
        if k > self.n or k_plus_1 > self.m:
            raise ValueError(f"Krylov subspace size {k} cannot exceed operator dimensions (m={self.m}, n={self.n})")

        if not np.isscalar(beta0):
             raise TypeError(f"beta0 must be a scalar norm, got {type(beta0)}")

        # Update iteration count
        self.iteration = Bk.shape[1]
        self.beta0= beta0

        # Compute SVD components of Bk
        self._compute_svd(Bk)
        
        # Compute the next regularization parameter
        res = self._compute_next_regalpha()

        # Handle return values
        if isinstance(res, (tuple, list)) and len(res) == 2:
            new_regalpha, func_value = res
        else:
            new_regalpha = res
            func_value = None
        
        # Update state if a new regularization parameter is found
        if new_regalpha is not None:
            self.regalpha = new_regalpha
            
            # Append to history
            self.regalpha_history.append(new_regalpha)
            # Verify func_value is scalar and not None or Nan or infinite
            if np.isscalar(func_value) and func_value is not None and not np.isnan(func_value) and not np.isinf(func_value):
                self.func_history.append(func_value)
            
            self._run_convergence_checks()
        self._print_status()
    
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

    def plot_function(self):
        '''Utility to plot the rule-specific function over a range of alphas.'''
        print("Plotting function not implemented")
    
    def _projected_residual_norm_sq(self, reg):
        """
        Computes the squared projected residual norm $\|r_\alpha\|_2^2$.

        The residual norm in the $k$-th Krylov subspace is given by:
        .. math::
            \rho(\alpha)^2 = \sum_{i=1}^k \left( \frac{\alpha^2}{\sigma_i^2 + \alpha^2} (u_i^T \beta e_1) \right)^2 
            + (u_{k+1}^T \beta e_1)^2

        Parameters
        ----------
        reg : float
            The regularization parameter $\alpha$.

        Returns
        -------
        float
            Squared norm of the residual.
        """
        # filt = alpha^2 / (s^2 + alpha^2)
        filt = (reg**2) / (self.Sbsq + reg**2)
        
        # Contribution from the subspace singular values
        r1 = np.sum(np.square(filt * (self.beta0 * self.u1T.T[:-1])))
        
        # Contribution from the component orthogonal to the subspace
        r2 = np.square(self.beta0 * self.u1T.T[-1])
        
        return r1 + r2
    
    def _projected_solution_norm_sq(self, reg):
        """
        Computes the squared projected solution norm $\|x_\alpha\|_2^2$.

        The solution norm in the $k$-th Krylov subspace is given by:
        .. math::
            \eta(\alpha)^2 = \sum_{i=1}^k \left( \frac{\sigma_i}{\sigma_i^2 + \alpha^2} (u_i^T \beta e_1) \right)^2

        Parameters
        ----------
        reg : float
            The regularization parameter $\alpha$.

        Returns
        -------
        float
            Squared norm of the solution.
        """
        # Solution filter factor: s / (s^2 + alpha^2)
        # Derived from x = V * inv(S^2 + alpha^2*I) * S * U.T * b
        filt = self.Sb / (self.Sbsq + reg**2)
        
        # Only the first k components contribute to the solution norm
        sol_norm_sq = np.sum(np.square(filt * (self.beta0 * self.u1T.T[:-1])))
        
        return sol_norm_sq

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
        func_value : float, optional
            The computed function value at the new regularization parameter.
            If not provided, return new_regalpha.
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

    def _compute_svd(self, Bk):
        '''Computes the SVD components for the projection matrix Bk.'''
        Ub, Sb, _ = scipy.linalg.svd(Bk)
        self.Sb = Sb
        self.Sbsq = np.square(Sb)
        self.u1T = Ub[0, :]
        self.Sbmax = Sb[0]
        self.Sbmin = Sb[-1]

        # Set regalpha bounds for plotting and optimization/root-finding
        self.regalpha_low = 0.0
        self.regalpha_high = self.Sbmax

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
    discrep_noise_level : float
        The estimated norm of the noise in the right-hand side, :math:`\eta = \|e\|_2`.

    Note
    ----
    The discrepancy principle requires an accurate estimate of the noise level. 
    If :math:`\eta` is underestimated, the solution may be under-regularised; 
    if overestimated, the solution will be over-smoothed.

    """

    def __init__(self, regalpha_saturation_tol, m, n, discrep_noise_level=0.0):
        super().__init__(regalpha_saturation_tol, m, n)
        self.rule_type = "discrep"
        self.noise_level = discrep_noise_level
    
    def _compute_next_regalpha(self):
        """Finds alpha such that ||r_alpha|| = noise_level using Brent's method.
        
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

        return result.root if result.converged else self.regalpha

    def func(self, regalpha):
        """Discrepancy function: phi(alpha) = ||r_alpha||^2 - noise^2."""
        return self._projected_residual_norm_sq(regalpha) - self.noise_level**2
    
    def plot_function(self):
        """
        Plot the discrepancy function stored from the last evaluation.
        Call this after an iteration that used rule='discrepancy'.
        We need to see where it crosses zero.
        """
        regalpha_grid = np.geomspace(self.regalpha_low, self.regalpha_high, 80)
        funcvec = np.array([self.func(reg) for reg in regalpha_grid])

        plt.figure()
        plt.semilogx(
            regalpha_grid, funcvec
        )  # semi-log plot to see zero crossing better
        plt.axhline(0, color="gray", linestyle="--")
        plt.xlabel("Regularisation parameter")
        plt.ylabel("Discrepancy function")
        plt.title("Discrepancy function vs regularisation parameter")
        if self.regalpha is not None:
            plt.axvline(
                self.regalpha,
                color="r",
                linestyle="--",
                label=f"Selected alpha: {self.regalpha:.2e}",
            )
        plt.legend()
        plt.grid()
        plt.show()

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

    Attributes
    ----------
    curvature : float
        The maximum curvature value $\kappa$ found at the currently selected $\alpha$.
    curvature_history : np.ndarray
        History of maximum curvature values found at each iteration $k$.
    """
    def __init__(self, regalpha_saturation_tol, m, n):
        super().__init__(regalpha_saturation_tol, m, n)
        self.rule_type = "l-curve"

    def _compute_next_regalpha(self):
        return 0.0, 0.0
    def func(self, regalpha):
        return 0.0

class UpdateRegGCV(UpdateRegBase):
    def __init__(self, regalpha_saturation_tol, m, n, gcv_type='weighted'):
        super().__init__(regalpha_saturation_tol, m, n)
        if gcv_type not in ['weighted','adaptive-weighted', 'standard']:
            raise ValueError(f"gcv_type must be one of 'weighted', 'adaptive-weighted', or 'standard'. Got {gcv_type}")
        self.rule_type = f"{gcv_type} gcv"
    def _compute_next_regalpha(self):
        return 0.0, 0.0
    def func(self, regalpha):
        return 0.0
