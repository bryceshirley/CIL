#!/usr/bin/env python
"""
IRLS on the simulated sphere, with FISTA as the reference point.

Usage
-----
    python irls_demo.py                          # 2D, everything
    python irls_demo.py --data 3d                # the full volume (needs a GPU)
    python irls_demo.py --only forms             # just standard against block
    python irls_demo.py --outer 5 --points 3     # a short run
"""

import argparse
import os
from time import time

import numpy as np

SECTIONS = ('regularisers', 'forms')

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--data', choices=('2d', '3d'), default='2d',
                    help='central slice or the full volume (default: 2d)')
parser.add_argument('--only', nargs='+', choices=SECTIONS, metavar='SECTION',
                    help='run only these sections: ' + ', '.join(SECTIONS))
parser.add_argument('--skip', nargs='+', choices=SECTIONS, default=(),
                    metavar='SECTION', help='run everything except these')
parser.add_argument('--outer', type=int,
                    help='IRLS outer iterations (default: 20 in 2D, 10 in 3D)')
parser.add_argument('--inner', type=int, default=10,
                    help='IRLS inner iterations per outer (default: 10)')
parser.add_argument('--fista', type=int,
                    help='FISTA iterations (default: outer x inner, so both '
                         'methods get the same number of applications of A)')
parser.add_argument('--angle-step', type=int, default=5,
                    help='angular subsampling: 5 keeps 60 of the 300 '
                         'projections')
parser.add_argument('--points', type=int, default=7,
                    help='how many iteration counts to sample each convergence '
                         'curve at (default: 7)')
parser.add_argument('--no-convergence', action='store_true',
                    help='skip the convergence curves: run each method once at '
                         'the full budget instead of once per sample point, '
                         'which is about three times faster and gives the '
                         'tables and the images only')
parser.add_argument('--alpha-scale', type=float, default=1.0,
                    help='multiply every alpha by this. The built-in values '
                         'came from denser data; sparser angles want more '
                         'regularisation')
parser.add_argument('--no-plots', action='store_true')
args = parser.parse_args()

import matplotlib                                                  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402

from cil.optimisation.algorithms import CGLS, FISTA, IRLS, LSQR    # noqa: E402
from cil.optimisation.functions import (L1Norm, L1Sparsity,        # noqa: E402
                                        LeastSquares, TotalVariation)
from cil.optimisation.operators import GradientOperator            # noqa: E402

from sphere_data import centre_slice, load_sphere, wavelet         # noqa: E402

sections = [s for s in (args.only or SECTIONS) if s not in args.skip]
OUTER = args.outer or (20 if args.data == '2d' else 10)
FISTA_ITERATIONS = args.fista or OUTER * args.inner
RESULTS = 'result_irls_demo_{}'.format(args.data)


def banner(text):
    print('\n' + '=' * 78 + '\n' + text + '\n' + '=' * 78)


banner('Simulated sphere ({}): IRLS {} outer x {} inner, FISTA {}'
       .format(args.data.upper(), OUTER, args.inner, FISTA_ITERATIONS))

data, A, ig, ground_truth, fdk = load_sphere(args.data, args.angle_step)
norm_A = A.norm()
truth_panel = centre_slice(ground_truth)

# One grey window for every figure in the run, taken from the FDK
# reconstruction: it is the widest thing on show, it is the reference everything
# else is trying to improve on, and fixing it once means panels are comparable
# across figures and not just within one. The percentiles rather than the raw
# range because a handful of extreme streak voxels would otherwise compress
# everything else into the middle of the grey scale.
FDK_WINDOW = tuple(np.percentile(centre_slice(fdk), [0.5, 99.5]))

# Every method starts from the FDK reconstruction, not from zero, so that they
# all begin at the same objective and the same error and the curves can be read
# against each other from their first point. Starting from zero makes the first
# step of each method a different-sized correction of the same trivial iterate,
# which shows up as an early separation that says nothing about the methods.
# The standard form is fine with a non-zero start: the inner solver maps it
# through reg_operator.direct into the transformed variable.
initial = fdk.copy()

# (name, L, alpha, g), where L is the regularising operator -- None for the
# identity, making the penalty a plain L1 on the image itself -- and g(alpha) is
# the function whose prox FISTA needs in order to be minimising the *same*
# thing. The alphas are the ones the two original drivers settled on.
#
# ``LeastSquares(A, b)`` is ``||Au-b||^2``, so each g carries alpha *squared*.
# Getting that wrong still produces a plausible-looking picture, of the wrong
# problem.
#
# The three proxes:
#   L1Norm         soft thresholding, exact.
#   L1Sparsity(Q)  Q^T soft(Q u), exact, and only exact because periodisation
#                  makes the wavelet transform orthogonal.
#   TotalVariation with isotropic=False, whose MixedL11Norm is exactly
#                  ``sum |grad u|`` -- the same anisotropic penalty IRLS
#                  reweights. Isotropic TV, the default, would be a different
#                  function and the two columns would not compare. This prox is
#                  itself iterative (FGP), so it is only approximate, and it is
#                  the reason the FISTA TV curve costs more per iteration.
haar = wavelet(ig, 'haar')
db4 = wavelet(ig, 'db4')
gradient = GradientOperator(ig)

REGULARISERS = [
    (name, L, alpha * args.alpha_scale, g) for name, L, alpha, g in [
        ('L1', None, 1.0, lambda a: a ** 2 * L1Norm()),
        ('Haar', haar, 1.5, lambda a: a ** 2 * L1Sparsity(haar)),
        ('Db4', db4, 1.5, lambda a: a ** 2 * L1Sparsity(db4)),
        # TV carries more than the others: sum |grad u| over a piecewise
        # constant phantom is a far smaller number than sum |W u| or sum |u|,
        # so at a comparable alpha the penalty barely bites and the TV row
        # reduces to an under-regularised least squares.
        ('TV', gradient, 6.0,
         lambda a: a ** 2 * TotalVariation(isotropic=False)),
    ]
]

SOLVERS = (LSQR, CGLS)

if not args.no_plots:
    os.makedirs(RESULTS, exist_ok=True)


# --------------------------------------------------------------------------- #
# the shared objective, and recording it
# --------------------------------------------------------------------------- #

def l1(x):
    """Sum of absolute values, recursing into a BlockDataContainer."""
    if hasattr(x, 'containers'):
        return sum(l1(c) for c in x.containers)
    return float(np.abs(x.as_array()).sum())


def make_objective(L, alpha):
    """``||Au-b||^2 + alpha^2 ||Lu||_1``, with L = I for ``None``."""
    def objective(u):
        residual = A.direct(u) - data
        return (float(residual.squared_norm())
                + alpha ** 2 * l1(u if L is None else L.direct(u)))
    return objective


def rmse(u):
    """Relative RMSE against the ground truth, over the whole container."""
    return float((u - ground_truth).norm() / ground_truth.norm())


class Trace:
    """A convergence curve: iteration count, wall clock, objective, RMSE."""

    def __init__(self, label):
        self.label = label
        self.iterations, self.times, self.values, self.rmses = [], [], [], []

    @property
    def rmse(self):
        return self.rmses[-1]


def budgets(total, points):
    """Roughly log-spaced iteration counts up to ``total``, ``total`` included."""
    if args.no_convergence:
        return [total]
    grid = np.geomspace(1, max(total, 1), max(points, 2))
    return [int(n) for n in np.unique(np.round(grid).astype(int))]


def sweep(build, total, objective, label):
    """
    Time a convergence curve by restarting, not by sampling.

    Each point is an independent run: build the algorithm fresh, start the
    clock, run exactly n iterations with nothing attached to it, stop the
    clock, and only then score the result. So the time reported against n
    iterations is the time to do those n iterations and nothing else.

    The obvious alternative -- one run with a callback recording the objective
    as it goes -- puts a forward projection inside the region being timed, and
    charges it unevenly: the callback fires once per iteration, and an IRLS
    outer iteration is a whole inner solve where a FISTA iteration is a single
    gradient step, so FISTA gets scored several times as often over a run of
    equal length. Deducting the measured overhead patches that, but the result
    is a time nobody could stopwatch. Restarting costs more -- the whole curve
    is about three times one run -- and buys numbers that mean what they say.

    ``build`` is a zero-argument factory, called once per point, because an
    algorithm that has already run cannot be rewound.
    """
    # Seed every curve with the shared starting iterate at zero iterations and
    # zero time. Without it the first plotted point is after one iteration, and
    # an IRLS iteration is a whole inner solve where a FISTA iteration is one
    # gradient step -- so the curves would appear to start from different
    # places when in fact they start from the same one.
    trace, solution = Trace(label), None
    trace.iterations.append(0)
    trace.times.append(0.0)
    trace.values.append(objective(initial))
    trace.rmses.append(rmse(initial))

    for n in budgets(total, args.points):
        algorithm = build()
        start = time()
        algorithm.run(n, verbose=0)
        elapsed = time() - start
        solution = algorithm.get_output().copy()
        trace.iterations.append(n)
        trace.times.append(elapsed)
        trace.values.append(objective(solution))
        trace.rmses.append(rmse(solution))
    print('  {:<34} {:7.2f}s  objective {:.5e}  rmse {:.4e}'
          .format(label, trace.times[-1], trace.values[-1], trace.rmse))
    return trace, solution


def build_irls(solver_class, L, alpha, form='auto'):
    """IRLS around an inner Krylov solve of the reweighted problem."""
    inner = solver_class(initial=initial, operator=A, data=data, alpha=alpha,
                         struct_operator=L, form=form, weighted=True)
    return IRLS(inner_solver=inner, max_inner_iteration=args.inner)


def build_fista(g, alpha):
    """FISTA on the same objective; ``g`` comes from ``REGULARISERS``."""
    return FISTA(initial=initial, f=LeastSquares(A, data), g=g(alpha),
                 step_size=1.0 / (2.0 * norm_A ** 2))


# --------------------------------------------------------------------------- #
# figures and tables
# --------------------------------------------------------------------------- #

def convergence_plot(traces, name, title):
    """Objective against wall clock and against iteration, side by side."""
    if args.no_plots or args.no_convergence or not traces:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Distinct dash patterns as well as colours: LSQR and CGLS often lie on top
    # of each other, and a curve hidden under another reads as a missing one.
    dashes = ['-', '--', ':', '-.']
    for index, trace in enumerate(traces):
        style = dashes[index % len(dashes)]
        # Markers, because each point is a run and the line between two of them
        # is interpolation rather than anything that was measured.
        axes[0].semilogy(trace.times, trace.values, style, marker='o',
                         markersize=3.5, label=trace.label)
        axes[1].semilogy(trace.iterations, trace.values, style, marker='o',
                         markersize=3.5, label=trace.label)
    axes[0].set_xlabel('wall clock (s)')
    axes[1].set_xlabel('iteration')
    for axis in axes:
        axis.set_ylabel(r'$\|Au-b\|^2 + \alpha^2\|Lu\|_1$')
        axis.grid(True, which='both', alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS, name), dpi=150, bbox_inches='tight')
    plt.close(figure)


def image_grid(panels, name, title):
    """
    One row of reconstructions in grey, all on the FDK window.

    Every figure uses ``FDK_WINDOW``, so a panel here can be held against a
    panel in any other figure in the run and the grey levels mean the same
    thing. Deriving the window from each figure's own contents instead would
    rescale it whenever the contents changed, which quietly turns "this
    reconstruction is darker" into "this figure happened to contain something
    brighter".

    The window is not clamped at zero. Clamping hides the undershoot at the
    edges and the negative half of the streaks, and makes two reconstructions
    that differ only below zero look identical. With a negative ``vmin`` the
    zero background sits at dark grey rather than black, so anything below zero
    is visibly darker than the background.

    One figure per regulariser rather than one for the whole run: at four
    regularisers times three methods the combined grid was too wide to read
    anything off.
    """
    if args.no_plots or not panels:
        return
    figure, axes = plt.subplots(1, len(panels), squeeze=False,
                                figsize=(3.0 * len(panels), 3.4))
    low, high = FDK_WINDOW
    for column, (label, array) in enumerate(panels):
        image = axes[0][column].imshow(array, cmap='gray', vmin=low, vmax=high)
        axes[0][column].set_title(label, fontsize=9)
        axes[0][column].axis('off')
    figure.colorbar(image, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
    figure.suptitle(title, y=1.02)

    figure.savefig(os.path.join(RESULTS, name), dpi=150, bbox_inches='tight')
    plt.close(figure)


def table(rows, columns):
    widths = [max(len(str(row[i])) for row in [columns] + rows)
              for i in range(len(columns))]
    line = '  '.join('{:<%d}' % width for width in widths)
    print('\n' + line.format(*columns))
    print('  '.join('-' * width for width in widths))
    for entry in rows:
        print(line.format(*(str(cell) for cell in entry)))


def row(*labels_and_trace):
    """A table row: some labels, then the numbers from a trace."""
    *labels, trace = labels_and_trace
    return labels + ['{:.5e}'.format(trace.values[-1]),
                     '{:.4e}'.format(trace.rmse),
                     '{:.2f}'.format(trace.times[-1])]


# --------------------------------------------------------------------------- #
# section 1 -- the regularisers, with FISTA alongside
# --------------------------------------------------------------------------- #

def section_regularisers():
    banner('Regularisers: IRLS against FISTA on the same objective')
    rows = []

    for name, L, alpha, g in REGULARISERS:
        print('\n--- {} (alpha = {:g}) ---'.format(name, alpha))
        objective = make_objective(L, alpha)
        traces = []
        panels = [('Ground truth', truth_panel), ('FDK', centre_slice(fdk))]

        for solver_class in SOLVERS:
            label = 'IRLS + {}'.format(solver_class.__name__)
            trace, solution = sweep(
                lambda s=solver_class: build_irls(s, L, alpha),
                OUTER, objective, '{} ({})'.format(label, name))
            traces.append(trace)
            rows.append(row(name, '{:g}'.format(alpha), label, trace))
            panels.append((label, centre_slice(solution)))

        trace, solution = sweep(lambda: build_fista(g, alpha),
                                FISTA_ITERATIONS, objective,
                                'FISTA ({})'.format(name))
        traces.append(trace)
        rows.append(row(name, '{:g}'.format(alpha), 'FISTA', trace))
        panels.append(('FISTA', centre_slice(solution)))

        convergence_plot(traces, 'convergence_{}.png'.format(name.lower()),
                         '{}: IRLS against FISTA, alpha = {:g}'
                         .format(name, alpha))
        image_grid(panels, 'images_{}.png'.format(name.lower()),
                   '{} ({}), alpha = {:g}'.format(name, args.data.upper(),
                                                  alpha))

    table(rows, ['regulariser', 'alpha', 'method', 'objective', 'rmse',
                 'wall clock (s)'])
    print('\nEvery row minimises the same function, so the objective column\n'
          'compares directly. Where FISTA reaches a lower value it is not\n'
          'merely faster but heading somewhere better: IRLS minimises the\n'
          'tau-smoothed surrogate sum sqrt(|Lu|^2 + tau^2), whose fixed point\n'
          'is not the L1 minimiser.\n'
          '\n'
          'The budgets are matched on applications of A, which is what a\n'
          'projector charges for. One IRLS outer iteration is {inner} inner\n'
          'Krylov iterations at one direct and one adjoint each, so {outer}\n'
          'outer costs {total}; FISTA gets {fista} iterations at two each, so\n'
          '{fista_total}. What that budget does not count is the prox, and on\n'
          'the TV row the prox is the whole story: TotalVariation solves its\n'
          'own inner problem, so FISTA pays for an accurate proximal step\n'
          'where IRLS pays for a cheap reweight. Read the TV row as a\n'
          'difference in what a step buys, not in how fast a step is.'
          .format(inner=args.inner, outer=OUTER, total=2 * OUTER * args.inner,
                  fista=FISTA_ITERATIONS, fista_total=2 * FISTA_ITERATIONS))


# --------------------------------------------------------------------------- #
# section 2 -- standard against block
# --------------------------------------------------------------------------- #

def section_forms():
    banner('Standard against block, for the orthogonal regularisers')
    print('\nThe same regularised problem, posed two ways:\n'
          '  block     K = [A ; alpha W L]   keeps the block structure\n'
          '  standard  K = A (W L)^-1        eliminates it\n'
          'Only an invertible L admits the standard form, so this is the\n'
          'wavelets and plain L1, not TV. LSQR appears on the block form only:\n'
          'in standard form it cannot warm start, IRLS resets it every outer\n'
          'iteration, and the curve would measure the restart rather than the\n'
          'form.')

    # LSQR is absent from the standard column on purpose. LSQR in standard form
    # cannot warm start -- it damps the step using the whole iterate, so a
    # carried-over state is wrong once the weights change -- and IRLS forces
    # reset_state=True for it. Every outer iteration would then restart from
    # zero, and the resulting curve says something about restarting, not about
    # the standard form. See IRLS.py, where the warning is raised.
    combinations = [(solver_class, form)
                    for solver_class in SOLVERS for form in ('block', 'standard')
                    if not (solver_class is LSQR and form == 'standard')]

    rows = []
    for name, L, alpha, _ in REGULARISERS:
        if L is not None and not L.is_orthogonal():
            continue
        print('\n--- {} (alpha = {:g}) ---'.format(name, alpha))
        objective = make_objective(L, alpha)
        traces, panels, solutions = [], [], {}

        for solver_class, form in combinations:
            label = '{} {}'.format(solver_class.__name__, form)
            trace, solution = sweep(
                lambda s=solver_class, f=form: build_irls(s, L, alpha, form=f),
                OUTER, objective, 'IRLS + {} ({})'.format(label, name))
            traces.append(trace)
            solutions[solver_class.__name__, form] = solution
            panels.append((label, centre_slice(solution)))
            rows.append(row(name, solver_class.__name__, form, trace))

        block, standard = solutions['CGLS', 'block'], solutions['CGLS', 'standard']
        difference = (block - standard).norm() / max(block.norm(), 1e-30)
        print('  under CGLS the two forms differ by {:.3e}, relative'
              .format(difference))
        rows.append([name, 'CGLS', 'relative difference',
                     '{:.3e}'.format(difference), '', ''])

        convergence_plot(traces, 'forms_{}.png'.format(name.lower()),
                         '{}: standard against block, alpha = {:g}'
                         .format(name, alpha))
        image_grid(panels, 'images_forms_{}.png'.format(name.lower()),
                   '{}: the two forms'.format(name))

    table(rows, ['regulariser', 'inner solver', 'form', 'objective', 'rmse',
                 'wall clock (s)'])
    print('\nThe forms pose the same problem, so at convergence they should\n'
          'agree, and the relative-difference rows say how far off that they\n'
          'are. They drift apart as tau falls: the weights spread, and the\n'
          "standard form's (WL)^-1 becomes badly conditioned -- the same\n"
          'mechanism that eventually stalls the outer loop. Read the two\n'
          'block rows against each other for the inner solver on its own: with\n'
          'warm starts available to both, LSQR and CGLS land in the same\n'
          'place, so the form is doing the work here, not the solver.')


RUNNERS = {'regularisers': section_regularisers, 'forms': section_forms}

started = time()
for section in SECTIONS:
    if section in sections:
        RUNNERS[section]()

banner('Done in {:.1f} s'.format(time() - started))
if not args.no_plots:
    print('figures: {}'.format(RESULTS))
