# CIL optimisation algorithms and linear operators
from cil.optimisation.algorithms import IRLS, LSQR, CGLS
from data_loader import load_and_process_sphere_2D
from cil.optimisation.operators import (
    WaveletOperator,
    GradientOperator,
    FiniteDifferenceOperator,
    MaskOperator,
    SymmetrisedGradientOperator,
    CompositionOperator,
)
import numpy as np

# CIL imports for data loading and visualisation
from cil.utilities.display import show2D

# Third-party imports
from time import time
import matplotlib.pyplot as plt
import os
from datetime import datetime


def get_exterior_mask_operator(ig, radius):
    """
    Creates a MaskOperator that applies 1s outside a circle of 'radius',
    and 0s inside.
    """
    shape = ig.shape
    center_y, center_x = shape[0] // 2, shape[1] // 2

    Y, X = np.ogrid[: shape[0], : shape[1]]
    dist_from_center = np.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
    mask_array = (dist_from_center > radius).astype(np.float32)

    mask_container = ig.allocate(0)
    mask_container.fill(mask_array)

    return MaskOperator(mask_container)


def get_hessian_op(ig):
    grad_op = GradientOperator(ig)
    sym_op = SymmetrisedGradientOperator(grad_op.range_geometry())
    return CompositionOperator(sym_op, grad_op)


# Load data
data, A, ig, ground_truth = load_and_process_sphere_2D(angle_step=5)

# Set up Initial
initial = A.domain_geometry().allocate(0)

# We set a tolerance, an initial guess, and a maximum number of iterations
maxit = 10
max_inner_iteration = 20

struct_op_and_regalpha = {
    "L1 Wavelet (Haar)": [WaveletOperator(ig, wavelet="haar"), 1.5],
    "L1 Wavelet (Db4)": [WaveletOperator(ig, wavelet="db4"), 1.5],
    "Horizontal TV": [FiniteDifferenceOperator(ig, direction=1), 5.0],
    "Vertical TV": [FiniteDifferenceOperator(ig, direction=0), 5.0],
    "TV": [GradientOperator(ig), 2.5],
    "L1": [None, 1.0],
    "Masked L1": [get_exterior_mask_operator(ig, radius=55), 3.0],
    "Hessian": [get_hessian_op(ig), 15.0],
}

# ---------------------------------------------------------
# Set up a single timestamped results folder
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"result_irls_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

print("=" * 60)
print(f"Saving all comparison outputs to directory: {results_dir}")
print("=" * 60)

# Iterate through the structural operators
for name, (struct_op, regalpha) in struct_op_and_regalpha.items():
    print(f"\n--- Evaluating {name} Regularisation (alpha={regalpha}) ---")

    # ---------------------------------------------------------
    # 1. Run LSQR
    # ---------------------------------------------------------
    print("Running LSQR...")
    lsqr = LSQR(
        operator=A,
        data=data,
        initial=initial.copy(),  # Pass a fresh copy just to be perfectly safe
        regalpha=regalpha,
        struct_operator=struct_op,
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
    )
    irls_cgls = IRLS(inner_solver=cgls, max_inner_iteration=max_inner_iteration)

    t_start = time()
    irls_cgls.run(maxit, verbose=True)
    t_cgls = time() - t_start
    print(f"CGLS Time taken: {t_cgls:.2f} seconds")

    out_cgls = irls_cgls.get_output()

    # ---------------------------------------------------------
    # 3. Plot Comparison (Ground Truth, LSQR, CGLS)
    # ---------------------------------------------------------
    show2D(
        [ground_truth, out_lsqr, out_cgls],
        title=[
            "Ground Truth",
            f"LSQR\nRegularisation: {name}, alpha: {regalpha:.2e}",
            f"CGLS\nRegularisation: {name}, alpha: {regalpha:.2e}",
        ],
        origin="upper",
        num_cols=3,
    )

    # Save the combined output
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    filename = os.path.join(results_dir, f"comparison_{safe_name}.png")

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"Saved visualization to {filename}")

    # Close the current figure so they don't consume memory across loop iterations
    plt.close()
