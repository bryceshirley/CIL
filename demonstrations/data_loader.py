# CIL core and io
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.io import ZEISSDataReader
from cil.io import TIFFStackReader

# CIL processors
from cil.processors import Binner, TransmissionAbsorptionConverter, Slicer

# CIL operators and algorithms
from cil.plugins.astra import ProjectionOperator

# CIL Utilities
from cil.utilities import dataexample

# Standard libraries
import numpy as np
import os
import json

DEVICE = "cpu"


def load_and_process_walnut(
    angle_step: int = 25,
    filename: str = "/mnt/share-private/materials/SIRF/Fully3D/CIL/Walnut/valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm",
):
    """
    Loads, preprocesses, and returns the walnut dataset and its Projection Operator.

    Parameters
    ----------
    angle_step : int, optional
        The step size for angular subsampling. Default is 25.
    filename : str, optional
        Path to the .txrm file.

    Returns
    -------
    data : DataContainer
        The processed CIL DataContainer.
    A : ProjectionOperator
        The Astra ProjectionOperator corresponding to the data geometry.
    ig : ImageGeometry
        The ImageGeometry of the dataset.
    """

    # --- 1. Load Data ---
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find file: {filename}")

    print(f"Loading data from: {filename}")
    reader = ZEISSDataReader()

    print("Setting up reader...")
    reader.set_up(file_name=filename)

    print("Reading data...")
    data3D = reader.read()

    # --- 2. Pre-processing ---
    print(f"Pre-processing with angle reduction step: {angle_step}...")

    # Reorder data to match Astra toolbox expectations
    data3D.reorder("astra")

    # Slicing: Select projections based on the requested step
    sliced_data = Slicer(roi={"angle": (0, 1601, angle_step)})(data3D)

    # Binning: Crop and bin the detector
    binned_data = Binner(
        roi={"horizontal": (120, -120, 2), "vertical": (120, -120, 2)}
    )(sliced_data)

    # Conversion: Transmission -> Absorption
    data = TransmissionAbsorptionConverter()(binned_data)

    # Artifact Removal
    background_val = np.mean(data.as_array()[80:100, 0:30])
    data -= background_val
    print(f"Artifact correction applied (background subtraction: {background_val:.4f})")

    # --- 3. Geometry Setup ---
    print("Updating Acquisition Geometry...")

    ag3D = data.geometry
    # Manually correct/set the angles
    ag3D.set_angles(ag3D.angles, initial_angle=0.2, angle_unit="radian")

    # Get Image Geometry
    ig3D = ag3D.get_ImageGeometry()

    # --- 4. Create Operator ---
    print("Creating ProjectionOperator...")
    try:
        A = ProjectionOperator(ig3D, ag3D, device=DEVICE)
    except Exception as e:
        raise RuntimeError(f"Failed to create Astra ProjectionOperator. \nError: {e}")

    print(f"Setup Complete. Data Shape: {data.shape}")

    return data, A, ig3D


def load_and_process_cylinder(
    dataset_no: int = 5, reduce_size: bool = True, filename: str = None
):
    """
    Reads processed data from TIFF files and a comma-separated angles file.

    Parameters
    ----------
    - dataset_no (int): The dataset identifier [1-7]. See Experimental Configurations below.
    - reduce_size (bool): Whether to reduce the size of the data
    - filename (str): The base path to the dataset

    Experimental Configurations
    ---------------------------

    The dataset consists of 7 different acquisition sets with varying exposure
    times and angular sampling rates.

    | Parameter             | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 | Dataset 5 | Dataset 6 | Dataset 7 |
    |-----------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
    |   Exposure Time (s)   | 7.5       | 15        | 30        | 60        | 60        | 60        | 60        |
    |   Number of Angles    | 840       | 420       | 210       | 105       | 210       | 240       | 840       |

    Returns
    -------
    - data (cil AcquisitionData): The processed projection data read from the TIFF files.
    - A (cil ProjectionOperator): The Astra ProjectionOperator corresponding to the data geometry.
    - ig (cil ImageGeometry): The ImageGeometry of the dataset.
    """
    # Validate dataset input
    if int(dataset_no) not in [1, 2, 3, 4, 5, 6, 7]:
        raise ValueError("dataset_no must be an integer between 1 and 7")

    # Map dataset number to exposure time and number of angles
    dataset_params = {
        1: (7.5, 840),
        2: (15, 420),
        3: (30, 210),
        4: (60, 105),
        5: (60, 210),
        6: (60, 240),
        7: (60, 840),
    }
    exposure_time, num_angles = dataset_params[int(dataset_no)]

    # convert exposure_time to string with '.' replaced with '_'
    exposure_time_str = str(exposure_time).replace(".", "_")
    # To update later:
    file_path = os.path.join(filename, f"exp_{exposure_time_str}_angles_{num_angles}")

    # Define paths for each type of data
    angles_file = os.path.join(file_path, "angles.csv")

    mi_file = os.path.join(file_path, "image.json")

    # Read angles from the CSV file
    angles = []
    with open(angles_file, "r") as csvfile:
        for line in csvfile:
            angles.append(float(line.strip()))
    if len(angles) != num_angles:
        raise ValueError(
            f"Number of angles in file ({len(angles)}) does not match expected number ({num_angles})"
        )

    # Read projection data using TIFFStackReader
    proj_reader = TIFFStackReader(file_name=file_path)
    data = proj_reader.read()

    # Read the JSON file
    with open(mi_file, "r") as jsonfile:
        d = json.load(jsonfile)
        # Extract last cor_tilt_finding
        final_cor_tilt = None
        for op in reversed(d["operation_history"]):
            if op["name"] == "cor_tilt_finding":
                final_cor_tilt = op["kwargs"]
                break

    rotation_centre = final_cor_tilt["rotation_centre"]
    tilt_angle_deg = final_cor_tilt["tilt_angle_deg"]

    offset = -rotation_centre + data.shape[2] / 2
    pixel_size = 48 * 10**-4
    offset = offset + np.tan(tilt_angle_deg * np.pi / 180) * (data.shape[1] / 2)
    pixel_size = 48 * 10**-4

    acquisition_geometry = (
        AcquisitionGeometry.create_Parallel3D()
        .set_panel(
            num_pixels=[data.shape[2], data.shape[1]],
            pixel_size=[pixel_size, pixel_size],
        )
        .set_angles(angles=angles)
    )
    ac_data = AcquisitionData(data, geometry=acquisition_geometry)

    ac_data.geometry.set_centre_of_rotation(
        -offset * pixel_size, angle=-tilt_angle_deg, angle_units="degree"
    )

    ac_data = TransmissionAbsorptionConverter()(ac_data)

    # reorder data to match default order for Astra operator
    ac_data.reorder("astra")

    if reduce_size:
        # Bin the data by a factor of 2 in all dimensions
        ac_data = Binner(
            roi={
                "horizontal": (None, None, 2),
                "vertical": (None, None, 2),
            }
        )(ac_data)
        ag = ac_data.geometry
        ig = ag.get_ImageGeometry()
        print(ig)

        # Re-centre the volume
        ig.voxel_num_x = 140
        ig.voxel_num_y = 140
        ig.voxel_num_z = 70
        ig.voxel_size_x = 2 * ig.voxel_size_x
        ig.voxel_size_y = 2 * ig.voxel_size_y
        ig.voxel_size_z = 2 * ig.voxel_size_z
        ig.center_z = -40 * ig.voxel_size_z
        ig.center_y = -5 * ig.voxel_size_y
        ig.center_x = +5 * ig.voxel_size_x
        print(ig)
    else:
        ag = ac_data.geometry
        ig = ag.get_ImageGeometry()
        print(ig)

    # Check geometry
    print(ac_data)
    return ac_data, ig, ag


def load_and_process_sphere_2D(angle_step: int = 5):
    """
    Loads and preprocesses the sphere dataset.

    Parameters
    ----------
    angle_step : int, optional
        The step size for angular subsampling. Default is 5.
    single_slice : bool, optional
        If True, extracts and processes only the central 2D slice.
        If False, processes the full 3D volume. Default is False.

    Returns
    -------
    absorption : DataContainer
        The processed CIL DataContainer (absorption data).
    A : ProjectionOperator
        The Astra ProjectionOperator corresponding to the data geometry.
    ig : ImageGeometry
        The ImageGeometry of the dataset.
    ground_truth : ImageData
        The ground truth volume/slice.
    """

    # Load data
    ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()
    data = dataexample.SIMULATED_CONE_BEAM_DATA.get()

    # Conditionally extract a single 2D slice
    data = data.get_slice(vertical="centre")
    ground_truth = ground_truth.get_slice(vertical="centre")

    # Convert to absorption and subsample angles
    absorption = TransmissionAbsorptionConverter()(data)
    absorption = Slicer(roi={"angle": (0, -1, angle_step)})(absorption)

    # --- ADD THIS INSTEAD ---
    # Reorder data to match Astra toolbox directly
    absorption.reorder("astra")
    ground_truth.reorder("astra")
    ig_astra = ground_truth.geometry

    # Setup Astra Projection Operator (Ensure device="cpu" is here!)
    A = ProjectionOperator(
        image_geometry=ig_astra, acquisition_geometry=absorption.geometry, device="cpu"
    )

    return absorption, A, ig_astra, ground_truth


# def load_and_process_sphere(angle_step: int = 5, single_slice: bool = False):
#     """
#     Loads and preprocesses the sphere dataset.

#     Parameters
#     ----------
#     angle_step : int, optional
#         The step size for angular subsampling. Default is 5.
#     single_slice : bool, optional
#         If True, extracts and processes only the central 2D slice.
#         If False, processes the full 3D volume. Default is False.

#     Returns
#     -------
#     absorption : DataContainer
#         The processed CIL DataContainer (absorption data).
#     A : ProjectionOperator
#         The Astra ProjectionOperator corresponding to the data geometry.
#     ig : ImageGeometry
#         The ImageGeometry of the dataset.
#     ground_truth : ImageData
#         The ground truth volume/slice.
#     recon : ImageData
#         The FDK reconstructed volume/slice.
#     """

#     # Load data
#     ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()
#     data = dataexample.SIMULATED_CONE_BEAM_DATA.get()

#     # Conditionally extract a single 2D slice
#     if single_slice:
#         data = data.get_slice(vertical='centre')
#         ground_truth = ground_truth.get_slice(vertical='centre')

#     # Convert to absorption and subsample angles
#     absorption = TransmissionAbsorptionConverter()(data)
#     absorption = Slicer(roi={'angle': (0, -1, angle_step)})(absorption)

#     # Reconstruct the initial guess using TIGRE's FDK
#     absorption.reorder('tigre')
#     ig_tigre = ground_truth.geometry
#     recon = FDK(absorption, image_geometry=ig_tigre).run()

#     # Reorder data to match Astra toolbox
#     absorption.reorder('astra')
#     ground_truth.reorder('astra')
#     recon.reorder('astra')

#     # Grab the updated geometry after reordering
#     ig_astra = ground_truth.geometry

#     # Visualization
#     if single_slice:
#         show2D([ground_truth, recon],
#                title=['Ground Truth', 'FDK Reconstruction'],
#                origin='upper', num_cols=2)
#     else:
#         # If 3D, grab the central slice just for the display plot
#         show2D([ground_truth.get_slice(vertical='centre'), recon.get_slice(vertical='centre')],
#                title=['Ground Truth (Central Slice)', 'FDK Reconstruction (Central Slice)'],
#                origin='upper', num_cols=2)

#     # Setup Astra Projection Operator
#     A = ProjectionOperator(image_geometry=ig_astra, acquisition_geometry=absorption.geometry, device=DEVICE)

#     return absorption, A, ig_astra, ground_truth, recon
