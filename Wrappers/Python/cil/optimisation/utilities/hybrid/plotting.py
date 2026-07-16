import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Any

from .maths import (
    find_optimal_alpha_via_grid,
    projected_residual_norm_sq,
    projected_solution_norm_sq,
    KrylovState,
)


def plot_rule_history(rule: Any, show_objective: bool = False, filepath: Optional[str] = None):
    """Utility to plot the history of regularization parameters."""
    history = rule.history.plot_data
    regalphas = history["regalphas"]

    if not regalphas:
        raise ValueError("No regularization parameter history available to plot.")

    num_subs = 2 if show_objective else 1
    fig, axes = plt.subplots(num_subs, 1, figsize=(8, 4 * num_subs), sharex=True)
    ax_alpha = axes[0] if show_objective else axes

    ax_alpha.plot(
        history["iterations"],
        regalphas,
        marker="o",
        color="tab:blue",
        label=r"$\alpha$",
    )
    ax_alpha.set_ylabel(r"Regularization $\alpha$")
    ax_alpha.set_yscale("log")
    ax_alpha.set_title(f"{rule.rule_type.upper()} Regularization Parameter History")
    ax_alpha.grid(True, which="both", ls="-", alpha=0.5)

    if show_objective:
        axes[1].plot(
            history["iterations"],
            history["objective_values"],
            marker="x",
            color="tab:red",
            linestyle="--",
        )
        axes[1].set_ylabel(f"{rule.rule_type.upper()} Function History")
        axes[1].set_xlabel("Iteration")
        axes[1].grid(True, alpha=0.5)
    else:
        ax_alpha.set_xlabel("Iteration")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()
    plt.close(fig)


def plot_rule_function(
    rule: Any,
    regalpha_limits: Optional[Tuple[float, float]] = None,
    num_points: int = 200,
    filepath: Optional[str] = None,
):
    """Router utility to plot the rule-specific function landscape."""
    rtype = rule.rule_type.lower()
    state = rule.return_krylov_state()
    upper = state.max_singular_value * rule.config.regalpha_high_multiplier
    bounds = regalpha_limits if regalpha_limits else (rule.config.eps, upper)

    # Define local proxy for objective to avoid lambdas
    def objective_proxy(alpha: float) -> float:
        return rule.evaluate_objective(alpha, state)

    if rtype == "discrep":
        _plot_discrep(rule, objective_proxy, bounds, num_points, filepath)
    elif "gcv" in rtype:
        _plot_gcv(rule, objective_proxy, bounds, num_points, filepath)
    elif rtype == "l-curve":
        _plot_lcurve(rule, state, bounds, num_points, filepath)
    elif rtype == "reginska":
        _plot_reginska(rule, state, bounds, num_points, filepath)
    elif rtype == "upre":
        _plot_upre(rule, state, objective_proxy, bounds, num_points, filepath)
    else:
        raise NotImplementedError(f"Plotting for {rtype} is not implemented.")


def _plot_discrep(rule, objective_func, bounds, pts, filepath):
    res = find_optimal_alpha_via_grid(objective_func, bounds=bounds, num_points=pts)
    res_norm_grid = np.sqrt(res.func_grid + rule.noise_level_estimate**2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(res.alpha_grid, res_norm_grid, label=r"Residual Norm $\|r_\alpha\|_2$")
    ax.semilogx(
        rule.current_regalpha,
        rule.noise_level_estimate,
        "ro",
        markersize=8,
        label=rf"$\alpha={rule.current_regalpha:.3e}$",
    )
    ax.axvline(rule.current_regalpha, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(
        rule.noise_level_estimate,
        color="green",
        linestyle="--",
        label=rf"Noise Level Estimate $\eta={rule.noise_level_estimate:.2e}$",
    )

    ax.set_xlabel(r"Regularisation parameter ($\alpha$)")
    ax.set_ylabel(r"$\|r_\alpha\|_2$")
    ax.set_title("Residual Norm vs. Target Noise")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()
    plt.close(fig)


def _plot_gcv(rule, objective_func, bounds, pts, filepath):
    res = find_optimal_alpha_via_grid(objective_func, bounds=bounds, num_points=pts)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(res.alpha_grid, res.func_grid, label=f"{rule.gcv_type.upper()} GCV Function")
    ax.semilogx(
        rule.current_regalpha,
        objective_func(rule.current_regalpha),
        "ro",
        markersize=8,
        label=rf"$\alpha={rule.current_regalpha:.3e}$",
    )
    ax.axvline(rule.current_regalpha, color="gray", linestyle=":", alpha=0.5)
    
    ax.set_xlabel(r"Regularisation parameter ($\alpha$)")
    ax.set_ylabel("GCV Function Value")
    ax.set_title(rf"{rule.gcv_type.upper()} GCV Function and Weight ($\omega={rule.omega:.3e}$)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()
    plt.close(fig)


def _plot_lcurve(rule, state, bounds, pts, filepath):
    alphas = np.geomspace(bounds[0], bounds[1], pts)
    r_vals_grid = [projected_residual_norm_sq(a, state, rule.b_norm) for a in alphas]
    x_vals_grid = [projected_solution_norm_sq(a, state, rule.b_norm) for a in alphas]

    fig = plt.figure(figsize=(12, 5))

    # Plot the L-curve
    ax = fig.add_subplot(1, 2, 1)
    ax.loglog(r_vals_grid, x_vals_grid, linestyle="-")
    ax.loglog(
        projected_residual_norm_sq(rule.current_regalpha, state, rule.b_norm),
        projected_solution_norm_sq(rule.current_regalpha, state, rule.b_norm),
        "ro",
        markersize=8,
        label=rf"$\alpha={rule.current_regalpha:.3e}$",
    )
    ax.set_xlabel(r"$\|B_k y(\alpha)-\|b\|_2 e_1\|_2$")
    ax.set_ylabel(r"$\|x(\alpha)\|_2$")
    ax.set_title("L-curve (projected)")
    ax.legend()

    # Plot the curvature function
    curv = [rule.evaluate_objective(a, state) for a in alphas]
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.semilogx(alphas, -np.array(curv), linestyle="-")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("Curvature")
    ax2.set_title("L-curve curvature (projected)")
    ax2.set_xlim(alphas[0], alphas[-1])
    ax2.semilogx(
        rule.current_regalpha,
        -rule.evaluate_objective(rule.current_regalpha, state),
        "ro",
        markersize=8,
        label=rf"$\alpha={rule.current_regalpha:.3e}$",
    )
    ax2.legend()
    ax2.axvline(rule.current_regalpha, color="gray", linestyle="--")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()
    plt.close(fig)


def _plot_reginska(rule, state, bounds, pts, filepath):
    alphas = np.geomspace(bounds[0], bounds[1], pts)
    r_vals_grid = [projected_residual_norm_sq(a, state, rule.b_norm) for a in alphas]
    x_vals_grid = [projected_solution_norm_sq(a, state, rule.b_norm) for a in alphas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the L-curve (Left)
    ax1.loglog(r_vals_grid, x_vals_grid, linestyle="-")
    ax1.loglog(
        projected_residual_norm_sq(rule.current_regalpha, state, rule.b_norm),
        projected_solution_norm_sq(rule.current_regalpha, state, rule.b_norm),
        "ro",
        markersize=8,
        label=rf"$\alpha={rule.current_regalpha:.3e}$",
    )
    ax1.set_xlabel(r"$\|B_k y(\alpha)-\|b\|_2 e_1\|_2$")
    ax1.set_ylabel(r"$\|x(\alpha)\|_2$")
    ax1.set_title("L-curve (projected)")
    ax1.legend()

    # Plot Reginska functional (Right)
    vals = [rule.evaluate_objective(a, state) for a in alphas]
    ax2.loglog(alphas, np.exp(vals), color="tab:red", label=r"$\Psi(\alpha)$")
    ax2.set_title(rf"Reginska Functional ($\mu={rule.mu}$)")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$\Psi(\alpha)$")
    ax2.semilogx(
        rule.current_regalpha,
        np.exp(rule.evaluate_objective(rule.current_regalpha, state)),
        "ro",
        markersize=8,
        label=rf"$\alpha={rule.current_regalpha:.3e}$",
    )
    ax2.legend()
    ax2.axvline(rule.current_regalpha, color="gray", linestyle=":")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()
    plt.close(fig)


def _plot_upre(rule, state, objective_func, bounds, pts, filepath):
    res = find_optimal_alpha_via_grid(objective_func, bounds=bounds, num_points=pts)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(res.alpha_grid, res.func_grid, label=r"UPRE $U(\alpha)$")
    ax.semilogx(
        rule.current_regalpha,
        objective_func(rule.current_regalpha),
        "ro",
        markersize=8,
        label=rf"$\alpha={rule.current_regalpha:.3e}$",
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$U(\alpha)$")
    ax.set_title("UPRE Function Minimization")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()
    plt.close(fig)