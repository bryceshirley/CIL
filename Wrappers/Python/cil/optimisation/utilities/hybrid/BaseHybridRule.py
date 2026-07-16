import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Pruned unused imports to keep the namespace clean
from .maths import KrylovState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleConfig:
    """Centralised configuration and magic numbers for regularisation rules."""

    eps: float = 1e-12
    regalpha_high_multiplier: float = 100.0
    svd_ratio_tol: float = 1e-6
    grad_tolerance_multiplier: float = 0.01
    default_grid_points: int = 200


@dataclass
class IterationHistory:
    """Encapsulates history tracking for matrix reconstruction and plotting."""

    # Krylov projection scalars
    gkb_alphas: List[float] = field(default_factory=list)
    gkb_betas: List[float] = field(default_factory=list)

    # Regularisation metrics
    regalphas: List[float] = field(default_factory=list)
    objective_values: List[float] = field(default_factory=list)

    def record_subspace_scalars(self, alpha: float, beta: float) -> None:
        """Records the bidiagonalization scalars for the current step."""
        self.gkb_alphas.append(alpha)
        self.gkb_betas.append(beta)

    def record_regularisation_metrics(self, regalpha: float, obj_val: float) -> None:
        """Records the computed regularisation parameter and objective function value."""
        self.regalphas.append(regalpha)
        self.objective_values.append(obj_val)

    def build_projected_operator(self) -> np.ndarray:
        """Constructs the lower bidiagonal Bk matrix from the current history."""
        k = len(self.gkb_alphas)
        Bk = np.zeros((k + 1, k))
        np.fill_diagonal(Bk, self.gkb_alphas)
        np.fill_diagonal(Bk[1:, :], self.gkb_betas[1:])
        return Bk

    @property
    def plot_data(self) -> Dict[str, List[float]]:
        """Convenience property to fetch histories specifically for plotting."""
        return {
            "iterations": list(range(1, len(self.regalphas) + 1)),
            "regalphas": self.regalphas,
            "objective_values": self.objective_values,
        }


@dataclass(frozen=True)
class StoppingState:
    """Output state returned at each iteration detailing the stopping rule's status."""

    converged: bool = False
    iteration: int = 0
    regalpha: float = 0.0


class BaseHybridRule(ABC):
    """Base class for updating regularisation parameters and determining stopping criteria."""

    def __init__(
        self,
        data_size: int,
        domain_size: int,
        tol: float = 1e-2,
        config: Optional[RuleConfig] = None,
    ):
        if data_size <= 0 or domain_size <= 0 or tol <= 0:
            raise ValueError(
                f"Invalid dimensions or tolerance: data={data_size}, domain={domain_size}, tol={tol}"
            )

        self.rule_type = "base-rule"
        self.m, self.n = data_size, domain_size
        self.tol = tol
        self.config = config or RuleConfig()
        self.history = IterationHistory()
        self.stopping_state = StoppingState()
        self.b_norm = 0.0
        self.current_regalpha = 0.0

    def reset_state(self, initial_alpha: float, initial_beta: float) -> None:
        """Resets the state for a new solver pass and seeds the initial subspace."""
        self.history = IterationHistory()
        self.history.record_subspace_scalars(initial_alpha, initial_beta)
        self.b_norm = initial_beta
        self.current_regalpha = 0.0
    
    def return_krylov_state(self) -> KrylovState:
        """Returns a KrylovState object built from the current history."""
        Bk = self.history.build_projected_operator()
        return KrylovState(bk=Bk)

    def update(self, alpha: float, beta: float) -> StoppingState:
        """Orchestrates a single iteration step."""
        
        # 1. Advance subspace history
        self.history.record_subspace_scalars(alpha, beta)
        state = self.return_krylov_state()

        lower_bound = 0.0
        upper_bound = state.max_singular_value * self.config.regalpha_high_multiplier

        # 2. Pass strictly typed locals into the optimization method
        new_regalpha = self._calculate_optimal_regalpha(state, lower_bound, upper_bound)
        
        if np.isfinite(new_regalpha):
            # Success: Update state and evaluate objective
            self.current_regalpha = new_regalpha
            obj_val = self.evaluate_objective(new_regalpha, state)
            
            # Safe cast for the objective value
            safe_obj = float(obj_val) if np.isfinite(obj_val) else np.nan
            self.history.record_regularisation_metrics(new_regalpha, safe_obj)
        else:
            # Failure: Pad the history arrays with NaNs
            self.history.record_regularisation_metrics(np.nan, np.nan)

        # 3. Check Convergence and Log
        converged = self._check_convergence()
        
        val_str = f"{self.current_regalpha:.4e}" if self.current_regalpha else "N/A"
        log.info(f"Iteration {state.iteration}: regalpha = {val_str} [{self.rule_type.upper()}]")

        self.stopping_state = StoppingState(
            converged=converged,
            iteration=state.iteration - 1,
            regalpha=self.current_regalpha,
        )

    def _check_convergence(self) -> bool:
        """Checks if the regularisation parameter has saturated based on tolerance."""
        if len(self.history.regalphas) < 2:
            return False

        prev = self.history.regalphas[-2]
        denom = abs(self.current_regalpha) + self.config.eps

        if (abs(self.current_regalpha - prev) / denom) < self.tol:
            log.debug(f"Alpha Saturation: Converged with regalpha={self.current_regalpha:.4e}")
            return True

        return False

    # -------------------------------------------------------------------------
    # Abstract Methods - Now completely pure and fully typed!
    # -------------------------------------------------------------------------

    @abstractmethod
    def _calculate_optimal_regalpha(
        self, state: KrylovState, lower_bound: float, upper_bound: float
    ) -> float:
        """Calculates the next optimal regularisation parameter."""
        pass
    
    @abstractmethod
    def evaluate_objective(self, regalpha: float, state: KrylovState) -> float:
        """Evaluates the objective function for a given regalpha."""
        pass
