# CIL optimisation algorithms and linear operators
from cil.optimisation.algorithms import IRLS, LSQR, CGLS
from data_loader import load_and_process_sphere
from cil.optimisation.operators import (
    WaveletOperator,
    GradientOperator,
)
import numpy as np

# CIL imports for data loading and visualisation
from cil.utilities.display import show2D

# Third-party imports
from time import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Load data
data, A, ig, ground_truth, fdk = load_and_process_sphere(angle_step=5)

# Set up Initial
initial = A.domain_geometry().allocate(0)

# We set a tolerance, an initial guess, and a maximum number of iterations
maxit = 10
max_inner_iteration = 20

# FIX: Added bnd_cond="periodization" to ensure perfect orthogonality
struct_op_and_regalpha = {
    "L1 Wavelet (Haar)": [WaveletOperator(ig, wavelet="haar", bnd_cond="periodization"), 1.0],
    "L1 Wavelet (Db4)": [WaveletOperator(ig, wavelet="db4", bnd_cond="periodization"), 1.0],
#     "TV": [GradientOperator(ig), 2.5],
#     "L1": [None, 1.0],
}


def plot_solutions(ground_truth, out_lsqr, out_cgls, fdk, name, regalpha, results_dir):
    show2D([ground_truth.get_slice(vertical='centre'), out_lsqr.get_slice(vertical='centre'), out_cgls.get_slice(vertical='centre'), fdk.get_slice(vertical='centre')],
               title=['Ground Truth (Central Slice)', f'LSQR (Central Slice)\nRegularisation: {name}, alpha: {regalpha:.2e}', f'CGLS (Central Slice)\nRegularisation: {name}, alpha: {regalpha:.2e}', 'FDK (Central Slice)'],
               origin='upper', num_cols=4)
    # Save the combined output
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    filename = os.path.join(results_dir, f"comparison_{safe_name}.png")

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"Saved visualization to {filename}")

    # Close the current figure so they don't consume memory across loop iterations
    plt.close()

# ---------------------------------------------------------
# Set up a single timestamped results folder
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"result_irls_sphere_3D_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

print("=" * 60)
print(f"Saving all comparison outputs to directory: {results_dir}")
print("=" * 60)

print(f"\n--- Evaluating algorithms with no regularisation ---")
print("Running LSQR...")
lsqr = LSQR(
    operator=A,
    data=data,
    initial=initial.copy(),  # Pass a fresh copy just to be perfectly safe
    regalpha=0.0
)
t_start = time()
lsqr.run(maxit*max_inner_iteration, verbose=True)
t_lsqr = time() - t_start
print(f"LSQR Time taken: {t_lsqr:.2f} seconds")

out_lsqr = lsqr.get_output()

print("\nRunning CGLS...")
cgls = CGLS(
    operator=A,
    data=data,
    initial=initial.copy(),
    regalpha=0.0,
)
t_start = time()
cgls.run(maxit*max_inner_iteration, verbose=True)
t_cgls = time() - t_start
print(f"CGLS Time taken: {t_cgls:.2f} seconds")

out_cgls = cgls.get_output()

plot_solutions(ground_truth, out_lsqr, out_cgls, fdk, 'None', 0.0, results_dir)


# Iterate through the structural operators
for name, (struct_op, regalpha) in struct_op_and_regalpha.items():
    for change_variable in [True, False]:
        print(f"\n--- Evaluating {name} Regularisation (alpha={regalpha}) ---")

        # ---------------------------------------------------------
        # 1. Run LSQR
        # ---------------------------------------------------------
        print("Running LSQR...")
        lsqr = LSQR(
            operator=A,
            data=data,
            initial=initial.copy(),  
            regalpha=regalpha,
            struct_operator=struct_op,
            change_variable=change_variable
        )
        irls_lsqr = IRLS(inner_solver=lsqr, max_inner_iteration=max_inner_iteration)

        t_start = time()
        irls_lsqr.run(maxit, verbose=True)
        t_lsqr = time() - t_start
        print(f"LSQR Time taken: {t_lsqr:.2f} seconds")

        out_lsqr = irls_lsqr.get_output()

        # ---------------------------------------------------------
        # 2. Run CGLS
        # ---------------------------------------------------------
        print("\nRunning CGLS...")
        cgls = CGLS(
            operator=A,
            data=data,
            initial=initial.copy(),
            regalpha=regalpha,
            struct_operator=struct_op,
            change_variable=change_variable
        )
        irls_cgls = IRLS(inner_solver=cgls, max_inner_iteration=max_inner_iteration)

        t_start = time()
        irls_cgls.run(maxit, verbose=True)
        t_cgls = time() - t_start
        print(f"CGLS Time taken: {t_cgls:.2f} seconds")

        out_cgls = irls_cgls.get_output()

        plot_solutions(ground_truth, out_lsqr, out_cgls, fdk, f"{name}_{change_variable}", regalpha, results_dir)