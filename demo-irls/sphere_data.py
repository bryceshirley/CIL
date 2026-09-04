"""
Loader for the simulated sphere, for the IRLS demos.

``data_loader.load_and_process_sphere`` is shared with the walnut and cylinder
work and carries their habits: it always pops a ``show2D`` window, so it cannot
be imported by anything headless, and it leaves the choice of regularising
operator -- and the boundary condition that choice needs -- to whoever imports
it, so every sphere demo repeats the same constructions and the one that
matters is easy to get wrong.

    from sphere_data import load_sphere, wavelet, centre_slice

    data, A, ig, ground_truth, fdk = load_sphere('2d')
"""

from cil.optimisation.operators import WaveletOperator
from cil.plugins.astra import ProjectionOperator
from cil.processors import Slicer, TransmissionAbsorptionConverter
from cil.recon import FDK
from cil.utilities import dataexample


def load_sphere(dimension='2d', angle_step=5):
    """
    Load the simulated sphere and build its forward operator.

    Parameters
    ----------
    dimension : {'2d', '3d'}
        ``'2d'`` takes the central slice, which runs on a CPU in seconds and is
        the right size for iterating on a demo. ``'3d'`` is the full 128^3
        volume and needs a GPU: Astra has no CPU fallback for 3D projections.
    angle_step : int
        Angular subsampling. The raw data has 300 projections, which is enough
        that FDK alone reconstructs the sphere well and there is nothing for a
        regulariser to demonstrate; the callers here pass a much coarser step.

    Returns
    -------
    (data, A, ig, ground_truth, fdk)
        Absorption data and a matching projector, both in Astra order, the
        image geometry, the truth the data was simulated from, and the filtered
        back-projection that every regularised solve is trying to beat.
    """
    dimension = dimension.lower()
    if dimension not in ('2d', '3d'):
        raise ValueError("dimension must be '2d' or '3d', not {!r}"
                         .format(dimension))

    ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()
    data = dataexample.SIMULATED_CONE_BEAM_DATA.get()

    if dimension == '2d':
        data = data.get_slice(vertical='centre')
        ground_truth = ground_truth.get_slice(vertical='centre')

    data = TransmissionAbsorptionConverter()(data)
    data = Slicer(roi={'angle': (0, -1, angle_step)})(data)

    # FDK through the TIGRE backend, which is the order the geometry is in at
    # this point; the reorder to Astra comes after, so that the projector and
    # the reconstruction agree.
    data.reorder('tigre')
    fdk = FDK(data, image_geometry=ground_truth.geometry).run(verbose=0)

    data.reorder('astra')
    ground_truth.reorder('astra')
    fdk.reorder('astra')

    ig = ground_truth.geometry
    A = ProjectionOperator(image_geometry=ig, acquisition_geometry=data.geometry)
    return data, A, ig, ground_truth, fdk


def wavelet(ig, wname='haar', level=2):
    r"""
    A wavelet operator that is safe to put under test.

    ``bnd_cond='periodization'`` rather than the ``'symmetric'`` default. The
    default fails ``LinearOperator.dot_test`` for every filter longer than
    haar -- direct and adjoint are not an adjoint pair -- so a solver built on
    it is solving something other than the stated problem. Periodisation also
    keeps the transform square, which is what makes it orthogonal and so what
    makes the standard form :math:`A(WL)^{-1}` and :class:`L1Sparsity`
    available at all. The sphere is 128 voxels on a side, so level 2 divides
    cleanly in both 2D and 3D.
    """
    return WaveletOperator(ig, wname=wname, level=level,
                           bnd_cond='periodization')


def centre_slice(x):
    """A 2D array to draw: the container itself in 2D, its centre slice in 3D."""
    array = x.as_array()
    return array if array.ndim == 2 else array[array.shape[0] // 2]
