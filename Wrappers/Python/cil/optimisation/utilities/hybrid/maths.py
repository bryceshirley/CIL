import numpy as np
import scipy.linalg
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from functools import cached_property

@dataclass(frozen=True)
class KrylovState:
    """Computes the SVD of the projected subspace immutably and lazily."""
    bk: np.ndarray
    
    @cached_property
    def _svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # full_matrices=True is explicitly required to compute the u1_tail component
        return scipy.linalg.svd(self.bk, full_matrices=True)

    @property
    def singular_values(self) -> np.ndarray:
        return self._svd[1]

    @property
    def u1(self) -> np.ndarray:
        return self._svd[0][0, :-1]

    @property
    def u1_tail(self) -> float:
        return self._svd[0][0, -1]

    @property
    def vt(self) -> np.ndarray:
        return self._svd[2]

    @cached_property
    def singular_values_squared(self) -> np.ndarray:
        return np.square(self.singular_values)

    @cached_property
    def u1_squared(self) -> np.ndarray:
        return np.square(self.u1)
        
    @property
    def max_singular_value(self) -> float:
        return self.singular_values[0]
        
    @property
    def min_singular_value(self) -> float:
        return self.singular_values[-1]

    @property
    def iteration(self) -> int:
        return len(self.singular_values)


@dataclass(frozen=True)
class GridSearchResult:
    """Encapsulates the results of a 1D grid search optimization."""
    best_alpha: float
    best_value: float
    func_grid: np.ndarray
    alpha_grid: np.ndarray


def find_optimal_alpha_via_grid(
    func: Callable[[float], float], 
    dfunc: Optional[Callable[[float], float]] = None, 
    bounds: Tuple[float, float] = (1e-12, 1e2), 
    num_points: int = 200
) -> GridSearchResult:
    """
    Pure functional 1D geometric grid search. 
    If a derivative function (dfunc) is provided, it isolates sign changes 
    (stationary points). Otherwise, it finds the global grid minimum.
    """
    start = max(bounds[0], 1e-12)
    stop = max(bounds[1], 1e-11)
    
    # Ensure a valid geometric range
    if stop <= start:
        stop = start * 10.0

    alpha_grid = np.geomspace(start, stop, num_points)
    func_grid = np.array([func(a) for a in alpha_grid])

    # Fallback to global minimum if no derivative function is provided
    if dfunc is None:
        best_idx = np.nanargmin(func_grid)
        return GridSearchResult(alpha_grid[best_idx], func_grid[best_idx], func_grid, alpha_grid)

    # Evaluate derivative and filter out invalid/non-finite values
    dphi = np.array([dfunc(a) for a in alpha_grid])
    valid_mask = np.isfinite(dphi)
    
    valid_alpha = alpha_grid[valid_mask]
    valid_dphi = dphi[valid_mask]
    valid_func = func_grid[valid_mask]

    # Locate indices where the derivative changes sign
    sign_changes = np.where(np.sign(valid_dphi[:-1]) != np.sign(valid_dphi[1:]))[0]

    if len(sign_changes) == 0:
        best_idx = np.argmin(valid_func)
        best_alpha = valid_alpha[best_idx]
    else:
        best_sign_change_idx = sign_changes[np.argmin(valid_func[sign_changes])]
        best_alpha = valid_alpha[best_sign_change_idx]

    return GridSearchResult(best_alpha, func(best_alpha), func_grid, alpha_grid)


# -------------------------------------------------------------------------
# Filter Factors & Solutions
# -------------------------------------------------------------------------

def residual_filter(reg: float, ks: KrylovState) -> np.ndarray:
    """
    Computes the residual filter factors for Tikhonov regularization.

    .. math::
        f_r(s,\alpha) = \frac{\alpha^2}{s^2 + \alpha^2}
    """
    reg_sq = reg ** 2
    return reg_sq / (ks.singular_values_squared + reg_sq)

def solution_filter(reg: float, ks: KrylovState) -> np.ndarray:
    """
    Computes the solution filter factors for Tikhonov regularization.

    .. math::
        f_x(s,\alpha) = \frac{s}{s^2 + \alpha^2}
    """
    return ks.singular_values / (ks.singular_values_squared + reg ** 2)

def compute_projected_solution(reg: float, ks: KrylovState, b_norm: float) -> np.ndarray:
    """Computes the regularized projected solution vector."""
    fx = solution_filter(reg, ks)
    return ks.vt.T @ (fx * (b_norm * ks.u1))


# -------------------------------------------------------------------------
# Projected Norms & Derivatives
# -------------------------------------------------------------------------

def projected_residual_norm_sq(reg: float, ks: KrylovState, b_norm: float) -> float:
    """Computes the squared projected residual norm."""
    fr = residual_filter(reg, ks)
    residual_k = np.sum(np.square(fr * (b_norm * ks.u1)))
    residual_tail = np.square(b_norm * ks.u1_tail)
    return float(residual_k + residual_tail)

def projected_solution_norm_sq(reg: float, ks: KrylovState, b_norm: float) -> float:
    """Computes the squared projected solution norm."""
    fx = solution_filter(reg, ks)
    return float(np.sum(np.square((b_norm * ks.u1) * fx)))

def projected_norm_first_derivatives(reg: float, ks: KrylovState, b_norm: float) -> Tuple[float, float]:
    """Compute analytical squared projected norms and their 1st derivatives."""
    fr = residual_filter(reg, ks)
    fx = solution_filter(reg, ks)
    fr_p, fx_p = filter_first_derivatives(reg, ks)

    b_norm_sq = b_norm ** 2
    R2_p = 2 * b_norm_sq * np.sum(ks.u1_squared * fr * fr_p)
    X2_p = 2 * b_norm_sq * np.sum(ks.u1_squared * fx * fx_p)

    return float(R2_p), float(X2_p)

def projected_norm_second_derivatives(reg: float, ks: KrylovState, b_norm: float) -> Tuple[float, float]:
    """Compute analytical squared projected norms and their 2nd derivatives."""
    fr = residual_filter(reg, ks)
    fx = solution_filter(reg, ks)
    fr_p, fx_p = filter_first_derivatives(reg, ks)
    fr_pp, fx_pp = filter_second_derivatives(reg, ks)

    b_norm_sq = b_norm ** 2
    R2_pp = 2 * b_norm_sq * np.sum(ks.u1_squared * (fr_p ** 2 + fr * fr_pp))
    X2_pp = 2 * b_norm_sq * np.sum(ks.u1_squared * (fx_p ** 2 + fx * fx_pp))

    return float(R2_pp), float(X2_pp)

def projected_norm_third_derivatives(reg: float, ks: KrylovState, b_norm: float) -> Tuple[float, float]:
    """Compute analytical third derivatives of squared projected residual and solution norms."""
    fr = residual_filter(reg, ks)
    fx = solution_filter(reg, ks)
    fr_p, fx_p = filter_first_derivatives(reg, ks)
    fr_pp, fx_pp = filter_second_derivatives(reg, ks)
    fr_ppp, fx_ppp = filter_third_derivatives(reg, ks)

    b_norm_sq = b_norm ** 2
    R2_ppp = 2 * b_norm_sq * np.sum(ks.u1_squared * (3 * fr_p * fr_pp + fr * fr_ppp))
    X2_ppp = 2 * b_norm_sq * np.sum(ks.u1_squared * (3 * fx_p * fx_pp + fx * fx_ppp))

    return float(R2_ppp), float(X2_ppp)


# -------------------------------------------------------------------------
# Filter Derivatives
# -------------------------------------------------------------------------

def filter_first_derivatives(reg: float, ks: KrylovState) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute first derivatives of Tikhonov filter factors.

    .. math::
        f_r' = \frac{2\alpha\sigma^2}{(\sigma^2 + \alpha^2)^2}, \quad
        f_x' = \frac{-2\alpha\sigma}{(\sigma^2 + \alpha^2)^2}
    """
    a2 = reg ** 2
    s2 = ks.singular_values_squared
    denom2 = np.square(s2 + a2)

    fr_p = 2 * reg * s2 / denom2
    fx_p = -2 * reg * ks.singular_values / denom2
    return fr_p, fx_p

def filter_second_derivatives(reg: float, ks: KrylovState) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute second derivatives of Tikhonov filter factors.

    .. math::
        f_r'' = \frac{2\sigma^2(\sigma^2 - 3\alpha^2)}{(\sigma^2 + \alpha^2)^3}, \quad
        f_x'' = \frac{2\sigma(3\alpha^2 - \sigma^2)}{(\sigma^2 + \alpha^2)^3}
    """
    a2 = reg ** 2
    s2 = ks.singular_values_squared
    denom3 = (s2 + a2) ** 3

    fr_pp = 2 * s2 * (s2 - 3 * a2) / denom3
    fx_pp = 2 * ks.singular_values * (3 * a2 - s2) / denom3
    return fr_pp, fx_pp

def filter_third_derivatives(reg: float, ks: KrylovState) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute third derivatives of Tikhonov filter factors.
    """
    a2 = reg ** 2
    s2 = ks.singular_values_squared
    s = ks.singular_values
    denom4 = (s2 + a2) ** 4
    
    fr_ppp = 24 * reg * s2 * (a2 - s2) / denom4
    fx_ppp = 24 * reg * s * (s2 - a2) / denom4

    return fr_ppp, fx_ppp