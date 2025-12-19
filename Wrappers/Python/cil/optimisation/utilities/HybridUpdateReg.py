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
        
        self.regalpha = None
        self.regalphavec = np.array([])
        self.converged = False
        self.iteration = 0
        self.EPS = 1e-15 

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

        self.iteration = Bk.shape[1]
        self._compute_svd(Bk)
        self.beta = beta0
        
        new_regalpha = self._compute_next_regalpha()
        
        if new_regalpha is not None:
            self.regalpha = new_regalpha
            self.regalphavec = np.append(self.regalphavec, new_regalpha)
            
            self._run_convergence_checks()
        self._print_status()
    
    def plot_history(self):
        '''Utility to plot the history of regularization parameters.'''
        plt.plot(self.regalphavec, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Regularization Parameter (alpha)')
        plt.title(f'History of {self.rule_type} Regularization Parameter')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

    def plot_function(self):
        '''Utility to plot the rule-specific function over a range of alphas.'''
        print("Plotting function not implemented")

    def _run_convergence_checks(self):
        '''Standard convergence check based on the saturation of alpha.
        
        Uses a relative change formula with a safety epsilon EPS.
        '''
        if len(self.regalphavec) < 2:
            return
        denom = abs(self.regalphavec[-1]) + self.EPS
        rel_change = abs(self.regalphavec[-1] - self.regalphavec[-2]) / denom
        
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

    def _compute_svd(self, Bk):
        '''Computes the SVD components for the projection matrix Bk.'''
        Ub, Sb, _ = scipy.linalg.svd(Bk, full_matrices=False)
        self.Sb = Sb
        self.Sbsq = np.square(Sb)
        self.u1T = Ub[0, :]
        self.Sbmax = Sb[0]
        self.Sbmin = Sb[-1]

        # Set regalpha bounds for plotting and optimization/root-finding
        self.reglow = 0.0
        self.reghigh = self.Sbmax

    def _print_status(self):
        '''Prints the current iteration status to the console.'''
        val = f"{self.regalpha:.4e}" if self.regalpha else "N/A"
        print(f"Iteration {self.iteration}: regalpha = {val} [{self.rule_type.upper()}]")