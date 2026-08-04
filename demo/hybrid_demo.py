# CIL optimisation algorithms and linear operators
from numpy import single
from cil.optimisation.algorithms import IRLS
from cil.optimisation.algorithms.LSQR_hybrid import LSQR
from cil.optimisation.utilities.hybrid import (
    ReginskaHybridRule,
    DiscrepHybridRule,
    LCurveHybridRule,
    GCVHybridRule,
    plot_rule_function,
    plot_rule_history,
)
from data_loader import load_and_process_sphere
from cil.optimisation.operators import WaveletOperator, PaddedDirichletGradientOperator
from matplotlib import pyplot as plt

# CIL imports for data loading and visualisation
from cil.utilities.display import show2D

# Third-party imports
from time import time
import os
from datetime import datetime

# ---------------------------------------------------------
# Set up timestamped results folder
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"result_hybrid_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Saving all outputs to directory: {results_dir}\n")

# Load data
data, A, ig, ground_truth, fdk = load_and_process_sphere(angle_step=5)

# Set up Initial
initial = A.domain_geometry().allocate(0)

# We set a tolerance, an initial guess, and a maximum number of iterations
maxit = 10
max_inner_iteration = 50
reset_state = False
struct_operators = {
    "L1": None,
    "Wavelet": WaveletOperator(ig, wavelet="haar"),
    "TV": PaddedDirichletGradientOperator(ig, backend="numpy"),
}

for name, W in struct_operators.items():
    current_rule = GCVHybridRule()

    print(
        f"Running LSQR with {current_rule.rule_type} stopping rule and {name} regularisation..."
    )

    lsqr = LSQR(
        operator=A,
        data=data,
        initial=initial,
        regalpha=current_rule,
        struct_operator=W,
    )
    irls = IRLS(
        inner_solver=lsqr,
        max_inner_iteration=max_inner_iteration,
        reset_state=reset_state,
    )

    t_start = time()
    irls.run(maxit, verbose=True)
    t_end = time()
    print(f"Time taken: {t_end - t_start:.2f} seconds")

    # Plot the history of the regularisation parameter and objective values
    history_filename = os.path.join(
        results_dir, f"lsqr_{current_rule.rule_type}_{name}_history.png"
    )
    plot_rule_history(current_rule, filepath=history_filename)

    # Plot the function landscape for the stopping rule
    function_filename = os.path.join(
        results_dir, f"lsqr_{current_rule.rule_type}_{name}_function.png"
    )
    plot_rule_function(current_rule, filepath=function_filename)

    # Use irls.get_output() to get the physical image, not the structure vector!
    show2D(
        [ground_truth, irls.get_output(), fdk],
        title=[
            "Ground Truth",
            f"{current_rule.rule_type}\n Optimal alpha: {lsqr.regalpha:.2e} Regularisation: {name}",
            "FDK"
        ],
        origin="upper",
        num_cols=3,
    )

    # Save the output
    # Strip parentheses and spaces for a cleaner filename
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    recon_filename = os.path.join(
        results_dir, f"lsqr_hybrid_reconstruction_{safe_name}.png"
    )
    plt.savefig(recon_filename, bbox_inches="tight", dpi=300)
    print(f"Saved visualization to {recon_filename}\n")

    # Close the current figure so they don't consume memory across loop iterations
    plt.close()
