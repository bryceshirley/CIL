#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import unittest
from contextlib import contextmanager

import numpy

from cil.framework import (AcquisitionGeometry, DataContainer, ImageGeometry,
                           VectorGeometry)
from cil.optimisation.algorithms import CGLS, FISTA, IRLS, LSQR
from cil.optimisation.functions import LeastSquares, TotalVariation
from cil.optimisation.operators import (GradientOperator, IdentityOperator,
                                        MatrixOperator, WaveletOperator)
from cil.optimisation.utilities.callbacks import Callback, IRLSEarlyStopping

from testclass import CCPiTestClass
from utils import initialise_tests

initialise_tests()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

@contextmanager
def count_allocated_elements():
    """
    Count the CIL containers allocated inside the block.

    A local copy of the helper in ``test_TikhonovOperator.py``, so that this
    file stands on its own. Only the leaf geometries are patched:
    ``BlockGeometry.allocate`` delegates to its children and allocates nothing
    itself. ``DataContainer.clone`` is patched too, because ``.copy()`` is an
    alias for it and neither goes through ``geometry.allocate``.
    """
    tally = {'elements': 0, 'containers': 0}
    saved = [(cls, 'allocate', cls.allocate)
             for cls in (ImageGeometry, VectorGeometry, AcquisitionGeometry)]
    saved.append((DataContainer, 'clone', DataContainer.clone))

    def wrap(original):
        def counted(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            tally['elements'] += int(numpy.prod(self.shape))
            tally['containers'] += 1
            return result
        return counted

    for cls, name, original in saved:
        setattr(cls, name, wrap(original))
    try:
        yield tally
    finally:
        for cls, name, original in saved:
            setattr(cls, name, original)


def small_least_squares(rows=8, columns=5, seed=17):
    """A dense over-determined problem small enough to check by hand."""
    rng = numpy.random.default_rng(seed)
    operator = MatrixOperator(rng.standard_normal((rows, columns)))
    data = operator.range_geometry().allocate(0)
    data.fill(rng.standard_normal(rows))
    return operator, data


def vector(geometry, values):
    """A container of `geometry` holding `values`."""
    out = geometry.allocate(0)
    out.fill(numpy.asarray(values, dtype=out.as_array().dtype))
    return out


def anisotropic_tv(image):
    """:math:`\\sum_i |(\\nabla u)_i|`, the penalty IRLS actually approximates.

    IRLS reweights every entry of ``Range(L)`` independently, so with
    :class:`GradientOperator` as :math:`L` the penalty being approximated is the
    *anisotropic* total variation -- the sum of the absolute values of both
    partial derivatives -- not the isotropic one, which couples them through a
    pointwise 2-norm. Comparing against FISTA means matching this, hence
    ``isotropic=False`` there.
    """
    gradient = GradientOperator(image.geometry)
    return sum(float(component.abs().sum())
               for component in gradient.direct(image).containers)


class RecordingCallback(Callback):
    """Records the outer iteration numbers it was called at."""

    def __init__(self):
        super().__init__(verbose=0)
        self.iterations = []

    def __call__(self, algorithm):
        self.iterations.append(algorithm.iteration)


class StubAlgorithm:
    """Hands out a prepared sequence of iterates, like an outer loop would."""

    def __init__(self, iterates):
        self.iterates = list(iterates)
        self.calls = 0

    def get_output(self, out=None):
        value = self.iterates[min(self.calls, len(self.iterates) - 1)]
        self.calls += 1
        if out is not None:
            out.fill(value)
            return out
        return value


# --------------------------------------------------------------------------- #
# the user interface for setting up regularisation
# --------------------------------------------------------------------------- #

class TestRegularisationInterface(CCPiTestClass):
    """
    The path a user actually takes: build an inner solver with a structural
    operator and an alpha, hand it to IRLS, run.
    """

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.operator = IdentityOperator(self.ig)
        self.data = self.ig.allocate('random', seed=41)
        self.alpha = 0.7

    def build(self, solver=LSQR, struct_operator=None, form='auto',
              weighted=True, **kwargs):
        return solver(operator=self.operator, data=self.data,
                      initial=self.ig.allocate(0), alpha=self.alpha,
                      struct_operator=struct_operator, form=form,
                      weighted=weighted, **kwargs)

    def test_l1_needs_no_structural_operator(self):
        """``struct_operator=None`` is plain L1 on the solution itself."""
        irls = IRLS(inner_solver=self.build(), max_inner_iteration=2)
        irls.run(2, verbose=0)
        self.assertTrue(numpy.isfinite(irls.get_output().as_array()).all())

    def test_total_variation_through_the_gradient(self):
        irls = IRLS(inner_solver=self.build(
                        struct_operator=GradientOperator(self.ig)),
                    max_inner_iteration=2)
        irls.run(2, verbose=0)
        self.assertTrue(numpy.isfinite(irls.get_output().as_array()).all())

    def test_wavelet_sparsity(self):
        wavelet = WaveletOperator(self.ig, wname='haar', level=1)
        irls = IRLS(inner_solver=self.build(struct_operator=wavelet),
                    max_inner_iteration=2)
        irls.run(2, verbose=0)
        self.assertTrue(numpy.isfinite(irls.get_output().as_array()).all())

    def test_both_inner_solvers_are_accepted(self):
        for solver in (LSQR, CGLS):
            with self.subTest(solver=solver.__name__):
                irls = IRLS(inner_solver=self.build(solver),
                            max_inner_iteration=2)
                irls.run(2, verbose=0)
                self.assertTrue(
                    numpy.isfinite(irls.get_output().as_array()).all())

    def test_weighted_solvers_choose_the_block_form(self):
        """
        ``weighted=True`` picks block even for an L that could go standard.

        The identity is square and orthogonal, so ``form='auto'`` would
        otherwise take the standard form, which cannot warm start under LSQR.
        """
        solver = self.build(struct_operator=IdentityOperator(self.ig))
        self.assertFalse(solver.standard_form)

    def test_lsqr_in_standard_form_is_forced_to_reset(self):
        solver = self.build(struct_operator=IdentityOperator(self.ig),
                            form='standard', weighted=False)
        self.assertTrue(solver.standard_form)
        with self.assertWarns(UserWarning):
            irls = IRLS(inner_solver=solver, max_inner_iteration=2)
        self.assertTrue(irls.reset_state)

    def test_cgls_warm_starts_in_either_form(self):
        solver = self.build(CGLS, struct_operator=IdentityOperator(self.ig),
                            form='standard', weighted=False)
        self.assertTrue(solver.supports_warm_start)
        irls = IRLS(inner_solver=solver, max_inner_iteration=2)
        self.assertFalse(irls.reset_state)

    def test_an_unregularised_solver_is_rejected(self):
        """alpha=0 returns the bare operator, which has no weights to set."""
        bare = LSQR(operator=self.operator, data=self.data,
                    initial=self.ig.allocate(0))
        with self.assertRaises(ValueError):
            IRLS(inner_solver=bare, max_inner_iteration=2)

    def test_weights_are_allocated_when_the_solver_did_not(self):
        solver = self.build(weighted=False)
        self.assertIsNone(solver.weights)
        IRLS(inner_solver=solver, max_inner_iteration=2)
        self.assertIsNotNone(solver.weights)


# --------------------------------------------------------------------------- #
# the maths, written out
# --------------------------------------------------------------------------- #

class TestWeightsAgainstTheFormula(CCPiTestClass):
    r"""
    The weight update is :math:`w = (|Lu|^2 + \tau^2)^{-1/4}`. These tests
    compute that in numpy, with no CIL operator in the way, and compare.
    """

    def setUp(self):
        self.operator, self.data = small_least_squares()
        self.geometry = self.operator.domain_geometry()
        self.alpha = 0.5
        self.tau = 0.3
        rng = numpy.random.default_rng(5)
        self.solution_array = rng.standard_normal(self.geometry.shape[0])

    def irls(self, struct_operator=None, tau=None, tau_factor=0.1):
        inner = LSQR(operator=self.operator, data=self.data,
                     initial=self.geometry.allocate(0), alpha=self.alpha,
                     struct_operator=struct_operator, weighted=True)
        return IRLS(inner_solver=inner, tau=self.tau if tau is None else tau,
                    tau_factor=tau_factor, max_inner_iteration=2)

    def test_identity_structure_reweights_the_solution_itself(self):
        irls = self.irls()
        solution = vector(self.geometry, self.solution_array)

        irls._update_weights(solution)

        expected = (self.solution_array ** 2 + self.tau ** 2) ** -0.25
        self.assertNumpyArrayAlmostEqual(
            expected, irls.inner_solver.operator.weights.as_array(), decimal=6)

    def test_a_structural_operator_reweights_its_range(self):
        r"""With :math:`L \neq I` the weights live in Range(L), on :math:`Lu`."""
        matrix = numpy.array([[1.0, -2.0, 0.0, 0.5, 0.0],
                              [0.0, 3.0, 1.0, 0.0, -1.0]])
        irls = self.irls(struct_operator=MatrixOperator(matrix))
        solution = vector(self.geometry, self.solution_array)

        irls._update_weights(solution)

        structure = matrix @ self.solution_array
        expected = (structure ** 2 + self.tau ** 2) ** -0.25
        self.assertNumpyArrayAlmostEqual(
            expected, irls.inner_solver.operator.weights.as_array(), decimal=6)

    def test_a_zero_solution_gives_the_tau_floor(self):
        """At u=0 every weight is :math:`\\tau^{-1/2}`, the largest it can be."""
        irls = self.irls()
        irls._update_weights(self.geometry.allocate(0))
        self.assertNumpyArrayAlmostEqual(
            numpy.full(self.geometry.shape[0], self.tau ** -0.5),
            irls.inner_solver.operator.weights.as_array(), decimal=6)

    def test_a_large_entry_is_barely_penalised(self):
        r"""
        As :math:`|Lu| \gg \tau` the weight tends to :math:`|Lu|^{-1/2}`, which
        is what turns the weighted 2-norm into the 1-norm:
        :math:`w^2 (Lu)^2 \to |Lu|`.
        """
        irls = self.irls(tau=1e-8)
        solution = vector(self.geometry, [100.0, 0.0, 0.0, 0.0, 0.0])
        irls._update_weights(solution)
        weight = irls.inner_solver.operator.weights.as_array()[0]
        self.assertAlmostEqual(100.0 ** -0.5, weight, places=6)
        self.assertAlmostEqual(100.0, (weight ** 2) * 100.0 ** 2, places=3)

    def test_tau_falls_by_its_factor_each_update(self):
        irls = self.irls(tau=1.0, tau_factor=0.25)
        solution = vector(self.geometry, self.solution_array)
        for expected in (0.25, 0.0625, 0.015625):
            irls._update_weights(solution)
            self.assertAlmostEqual(expected, irls.tau, places=12)

    def test_tau_stops_at_its_floor(self):
        irls = self.irls(tau=1e-7, tau_factor=0.001)
        solution = vector(self.geometry, self.solution_array)
        for _ in range(5):
            irls._update_weights(solution)
        self.assertEqual(1e-8, irls.tau)

    def test_the_weight_update_allocates_nothing(self):
        irls = self.irls()
        solution = vector(self.geometry, self.solution_array)
        irls._update_weights(solution)          # warm up any lazy state
        with count_allocated_elements() as tally:
            irls._update_weights(solution)
        self.assertEqual(0, tally['containers'])


class TestOneOuterIterationByHand(CCPiTestClass):
    """
    One outer iteration of IRLS, rebuilt step by step against a second solver
    driven by hand. Nothing here calls ``IRLS.update``; the point is that the
    sequence -- read the iterate, reweight, reset, run the inner solver -- is
    what the class does, and that it produces the same numbers.
    """

    def setUp(self):
        self.operator, self.data = small_least_squares()
        self.geometry = self.operator.domain_geometry()
        self.alpha = 0.4
        self.tau = 0.5
        self.inner_iterations = 4
        rng = numpy.random.default_rng(23)
        self.initial_array = rng.standard_normal(self.geometry.shape[0])

    def solver(self):
        return LSQR(operator=self.operator, data=self.data,
                    initial=vector(self.geometry, self.initial_array),
                    alpha=self.alpha, weighted=True)

    def test_one_iteration_matches_the_steps_written_out(self):
        irls = IRLS(inner_solver=self.solver(), tau=self.tau, tau_factor=0.1,
                    max_inner_iteration=self.inner_iterations)
        irls.run(1, verbose=0)

        # The same thing, by hand. run(1) performs exactly one update(),
        # because Algorithm.run adds an iteration at iteration == -1 and that
        # first __next__ only records the objective.
        by_hand = self.solver()
        solution = by_hand.get_output()                    # (1) read the iterate
        weights = (solution.as_array() ** 2                # (2) reweight
                   + self.tau ** 2) ** -0.25
        by_hand.operator.weights.fill(weights)
        by_hand.initial = solution                         # (3) warm start
        by_hand.initialise_variables()                     # (4) reset
        by_hand.run(self.inner_iterations, verbose=0)      # (5) inner solve

        self.assertNumpyArrayAlmostEqual(
            by_hand.get_output().as_array(),
            irls.get_output().as_array(), decimal=6)

    def test_the_first_weights_come_from_the_initial_guess(self):
        """
        The reweighting reads the iterate *before* the inner solver runs, so
        the weights of the first outer iteration are those of ``initial``.
        """
        irls = IRLS(inner_solver=self.solver(), tau=self.tau,
                    max_inner_iteration=self.inner_iterations)
        irls.run(1, verbose=0)

        expected = (self.initial_array ** 2 + self.tau ** 2) ** -0.25
        self.assertNumpyArrayAlmostEqual(
            expected, irls.inner_solver.operator.weights.as_array(), decimal=6)

    def test_reset_state_discards_the_warm_start(self):
        """
        With ``reset_state=True`` the inner solver restarts from the original
        initial each time, so its iterate differs from the warm-started one.
        """
        warm = IRLS(inner_solver=self.solver(), tau=self.tau,
                    max_inner_iteration=self.inner_iterations)
        cold = IRLS(inner_solver=self.solver(), tau=self.tau, reset_state=True,
                    max_inner_iteration=self.inner_iterations)
        warm.run(3, verbose=0)
        cold.run(3, verbose=0)
        self.assertFalse(numpy.allclose(warm.get_output().as_array(),
                                        cold.get_output().as_array()))

    def test_run_performs_exactly_the_requested_outer_iterations(self):
        irls = IRLS(inner_solver=self.solver(),
                    max_inner_iteration=self.inner_iterations)
        recorder = RecordingCallback()
        irls.run(4, callbacks=[recorder], verbose=0)
        # Callbacks fire after every __next__, including the one at iteration
        # -1 that only records the objective, so 0 is present and the real
        # outer iterations are 1..4.
        self.assertEqual([0, 1, 2, 3, 4], recorder.iterations)
        self.assertEqual(4, irls.iteration)


# --------------------------------------------------------------------------- #
# convergence
# --------------------------------------------------------------------------- #

class TestConvergence(CCPiTestClass):

    def test_outer_iterates_settle(self):
        """The relative change between outer iterates decreases towards zero."""
        operator, data = small_least_squares(rows=12, columns=6, seed=3)
        inner = LSQR(operator=operator, data=data,
                     initial=operator.domain_geometry().allocate(0),
                     alpha=0.3, weighted=True)
        irls = IRLS(inner_solver=inner, max_inner_iteration=10)

        changes, previous = [], None
        for _ in range(8):
            irls.run(1, verbose=0)
            current = irls.get_output().as_array().copy()
            if previous is not None and numpy.linalg.norm(previous):
                changes.append(float(numpy.linalg.norm(current - previous)
                                     / numpy.linalg.norm(previous)))
            previous = current

        self.assertLess(changes[-1], changes[0])
        self.assertLess(changes[-1], 1e-3)

    def test_sparse_recovery_beats_the_unregularised_solution(self):
        """
        The point of L1: on an under-determined system with a sparse truth,
        least squares returns the dense minimum-norm solution and IRLS does
        not.
        """
        rng = numpy.random.default_rng(11)
        rows, columns = 30, 60
        matrix = rng.standard_normal((rows, columns)) / numpy.sqrt(rows)
        truth = numpy.zeros(columns)
        truth[rng.choice(columns, size=5, replace=False)] = rng.standard_normal(5)

        operator = MatrixOperator(matrix)
        geometry = operator.domain_geometry()
        data = operator.range_geometry().allocate(0)
        data.fill(matrix @ truth)

        plain = LSQR(operator=operator, data=data,
                     initial=geometry.allocate(0))
        plain.run(200, verbose=0)

        inner = LSQR(operator=operator, data=data,
                     initial=geometry.allocate(0), alpha=0.05, weighted=True)
        irls = IRLS(inner_solver=inner, tau=1.0, tau_factor=0.5,
                    max_inner_iteration=50)
        irls.run(20, verbose=0)

        norm = numpy.linalg.norm(truth)
        plain_error = numpy.linalg.norm(
            plain.get_output().as_array() - truth) / norm
        irls_error = numpy.linalg.norm(
            irls.get_output().as_array() - truth) / norm

        self.assertLess(irls_error, plain_error)
        self.assertLess(irls_error, 0.25)

    def test_both_inner_solvers_reach_the_same_place(self):
        operator, data = small_least_squares(rows=14, columns=7, seed=8)
        geometry = operator.domain_geometry()
        outputs = {}
        for solver in (LSQR, CGLS):
            inner = solver(operator=operator, data=data,
                           initial=geometry.allocate(0), alpha=0.3,
                           weighted=True)
            irls = IRLS(inner_solver=inner, max_inner_iteration=20)
            irls.run(8, verbose=0)
            outputs[solver.__name__] = irls.get_output().as_array().copy()

        difference = numpy.linalg.norm(outputs['LSQR'] - outputs['CGLS'])
        self.assertLess(difference / numpy.linalg.norm(outputs['LSQR']), 1e-3)


class TestAgainstFISTATotalVariation(CCPiTestClass):
    """
    IRLS with the gradient as :math:`L` solves the same TV problem that FISTA
    solves with a :class:`TotalVariation` proximal term, so the two should land
    in the same place. ``isotropic=False`` because IRLS reweights each partial
    derivative independently, which is the anisotropic penalty.
    """

    def setUp(self):
        self.ig = ImageGeometry(16, 16)
        self.operator = IdentityOperator(self.ig)
        self.alpha = 0.3

        # A piecewise-constant phantom, which is what TV is for, plus noise.
        array = numpy.zeros((16, 16), dtype=numpy.float32)
        array[4:12, 4:12] = 1.0
        rng = numpy.random.default_rng(2)
        self.truth = self.ig.allocate(0)
        self.truth.fill(array)
        self.data = self.ig.allocate(0)
        self.data.fill(array + 0.1 * rng.standard_normal(array.shape))

    def objective(self, image):
        residual = float((self.operator.direct(image) - self.data).norm()) ** 2
        return residual + self.alpha ** 2 * anisotropic_tv(image)

    def test_irls_reaches_the_fista_objective(self):
        inner = LSQR(operator=self.operator, data=self.data,
                     initial=self.ig.allocate(0), alpha=self.alpha,
                     struct_operator=GradientOperator(self.ig), weighted=True)
        irls = IRLS(inner_solver=inner, max_inner_iteration=25)
        irls.run(15, verbose=0)

        fista = FISTA(initial=self.ig.allocate(0),
                      f=LeastSquares(self.operator, self.data, c=1.0),
                      g=(self.alpha ** 2) * TotalVariation(
                          max_iteration=100, isotropic=False))
        fista.run(300, verbose=0)

        irls_objective = self.objective(irls.get_output())
        fista_objective = self.objective(fista.get_output())

        # IRLS smooths the 1-norm with tau, so it cannot beat a true proximal
        # solver outright; it should get close from above.
        self.assertLess(irls_objective,
                        fista_objective * 1.05 + 1e-6)
        self.assertLess(self.objective(self.data) * 0.9, fista_objective * 3)

    def test_irls_denoises_towards_the_truth(self):
        inner = LSQR(operator=self.operator, data=self.data,
                     initial=self.ig.allocate(0), alpha=self.alpha,
                     struct_operator=GradientOperator(self.ig), weighted=True)
        irls = IRLS(inner_solver=inner, max_inner_iteration=25)
        irls.run(15, verbose=0)

        before = float((self.data - self.truth).norm())
        after = float((irls.get_output() - self.truth).norm())
        self.assertLess(after, before)


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #

class TestMemory(CCPiTestClass):
    """
    The three-tier contract seen from the outer loop: once ``set_up`` has run,
    an IRLS outer iteration allocates nothing, so the loop runs at constant
    memory however long it goes on.
    """

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.operator = IdentityOperator(self.ig)
        self.data = self.ig.allocate('random', seed=13)

    def inner(self, struct_operator=None, weighted=True, form='auto'):
        return LSQR(operator=self.operator, data=self.data,
                    initial=self.ig.allocate(0), alpha=0.5,
                    struct_operator=struct_operator, form=form,
                    weighted=weighted)

    def test_an_outer_iteration_allocates_nothing(self):
        irls = IRLS(inner_solver=self.inner(), max_inner_iteration=3)
        irls.run(1, verbose=0)                      # pay for any lazy set-up
        with count_allocated_elements() as tally:
            irls.run(3, verbose=0)
        self.assertEqual(0, tally['containers'])

    def test_a_structural_operator_costs_nothing_extra_per_iteration(self):
        irls = IRLS(inner_solver=self.inner(GradientOperator(self.ig)),
                    max_inner_iteration=3)
        irls.run(1, verbose=0)
        with count_allocated_elements() as tally:
            irls.run(3, verbose=0)
        self.assertEqual(0, tally['containers'])

    def test_standard_form_holds_one_extra_solution_buffer(self):
        """
        In standard form ``get_output`` maps back through :math:`(WL)^{-1}`,
        which needs somewhere to write. IRLS allocates that once, in its
        constructor, rather than on every outer iteration.
        """
        inner = self.inner(struct_operator=IdentityOperator(self.ig),
                           form='standard', weighted=False)
        with self.assertWarns(UserWarning):
            irls = IRLS(inner_solver=inner, max_inner_iteration=3)
        self.assertIsNotNone(irls.tmp_solution)

        irls.run(1, verbose=0)
        with count_allocated_elements() as tally:
            irls.run(3, verbose=0)
        self.assertEqual(0, tally['containers'])

    def test_attaching_to_an_unweighted_solver_costs_the_weights_once(self):
        """
        The documented exception to the contract: a solver built without
        ``weighted=True`` pays at attach time instead of in ``set_up``.

        With L the identity that is the weights in Range(L), and the
        accumulator the weighted adjoint then needs in the solution space --
        two containers of m, once, never repeated.
        """
        m = self.ig.shape[0] * self.ig.shape[1]
        inner = self.inner(weighted=False, form='block')
        with count_allocated_elements() as tally:
            IRLS(inner_solver=inner, max_inner_iteration=3)
        self.assertEqual(2, tally['containers'])
        self.assertEqual(2 * m, tally['elements'])

    def test_a_structural_operator_adds_its_staging_buffer(self):
        """
        A genuine two-stage :math:`WL` needs an intermediate in Range(L) as
        well, which the identity case collapses away. For the gradient on a
        square image Range(L) is 2m, so the weights and that intermediate come
        to 4m, spread over four leaf containers. The adjoint accumulator does
        not appear because a non-identity L already made ``set_up`` allocate
        it, which is the identity case's second container.
        """
        m = self.ig.shape[0] * self.ig.shape[1]
        inner = self.inner(GradientOperator(self.ig), weighted=False,
                           form='block')
        with count_allocated_elements() as tally:
            IRLS(inner_solver=inner, max_inner_iteration=3)
        self.assertEqual(4, tally['containers'])
        self.assertEqual(4 * m, tally['elements'])

    def test_a_weighted_solver_costs_nothing_at_attach_time(self):
        inner = self.inner(weighted=True)
        with count_allocated_elements() as tally:
            IRLS(inner_solver=inner, max_inner_iteration=3)
        self.assertEqual(0, tally['containers'])


# --------------------------------------------------------------------------- #
# the stopping condition
# --------------------------------------------------------------------------- #

class TestIRLSEarlyStopping(CCPiTestClass):

    def setUp(self):
        self.geometry = VectorGeometry(4)

    def iterate(self, values):
        return vector(self.geometry, values)

    def test_the_first_call_only_records(self):
        """
        Nothing to compare against at iteration -1, where ``__next__`` records
        the initial objective without running an update.
        """
        callback = IRLSEarlyStopping(epsilon=1.0, verbose=0)
        algorithm = StubAlgorithm([self.iterate([1.0, 1.0, 1.0, 1.0])])
        callback(algorithm)                      # must not raise
        self.assertEqual(numpy.inf, callback.change)

    def test_it_stops_once_the_iterates_settle(self):
        first = self.iterate([1.0, 0.0, 0.0, 0.0])
        second = self.iterate([1.0 + 1e-9, 0.0, 0.0, 0.0])
        callback = IRLSEarlyStopping(epsilon=1e-4, verbose=0)
        algorithm = StubAlgorithm([first, second])

        callback(algorithm)
        with self.assertRaises(StopIteration):
            callback(algorithm)

    def test_it_keeps_going_while_the_iterates_move(self):
        first = self.iterate([1.0, 0.0, 0.0, 0.0])
        second = self.iterate([2.0, 0.0, 0.0, 0.0])
        callback = IRLSEarlyStopping(epsilon=1e-4, verbose=0)
        algorithm = StubAlgorithm([first, second])

        callback(algorithm)
        callback(algorithm)                      # must not raise
        self.assertAlmostEqual(1.0, callback.change, places=6)

    def test_the_change_is_relative_to_the_previous_iterate(self):
        first = self.iterate([10.0, 0.0, 0.0, 0.0])
        second = self.iterate([11.0, 0.0, 0.0, 0.0])
        callback = IRLSEarlyStopping(epsilon=0.0, verbose=0)
        algorithm = StubAlgorithm([first, second])

        callback(algorithm)
        callback(algorithm)
        self.assertAlmostEqual(0.1, callback.change, places=6)

    def test_a_zero_iterate_does_not_divide_by_zero(self):
        callback = IRLSEarlyStopping(epsilon=1e-4, verbose=0)
        algorithm = StubAlgorithm([self.iterate([0.0, 0.0, 0.0, 0.0]),
                                   self.iterate([1.0, 0.0, 0.0, 0.0])])
        callback(algorithm)
        callback(algorithm)                      # must not raise
        self.assertEqual(numpy.inf, callback.change)

    def test_the_scratch_is_released_when_out_is_ignored(self):
        """
        Block form hands back the live iterate rather than filling ``out``, so
        the buffer is dead weight and the callback drops it.
        """
        class BlockFormAlgorithm(StubAlgorithm):
            def get_output(self, out=None):
                value = self.iterates[min(self.calls,
                                          len(self.iterates) - 1)]
                self.calls += 1
                return value                     # ignores `out`

        callback = IRLSEarlyStopping(epsilon=0.0, verbose=0)
        algorithm = BlockFormAlgorithm([self.iterate([1.0, 0.0, 0.0, 0.0]),
                                        self.iterate([2.0, 0.0, 0.0, 0.0])])
        callback(algorithm)
        self.assertIsNotNone(callback.scratch)
        callback(algorithm)
        self.assertIsNone(callback.scratch)

    def test_the_scratch_is_kept_when_out_is_honoured(self):
        callback = IRLSEarlyStopping(epsilon=0.0, verbose=0)
        algorithm = StubAlgorithm([self.iterate([1.0, 0.0, 0.0, 0.0]),
                                   self.iterate([2.0, 0.0, 0.0, 0.0])])
        callback(algorithm)
        callback(algorithm)
        self.assertIsNotNone(callback.scratch)

    def test_comparing_costs_no_allocation_once_running(self):
        callback = IRLSEarlyStopping(epsilon=0.0, verbose=0)
        algorithm = StubAlgorithm([self.iterate([1.0, 0.0, 0.0, 0.0]),
                                   self.iterate([2.0, 0.0, 0.0, 0.0]),
                                   self.iterate([3.0, 0.0, 0.0, 0.0])])
        callback(algorithm)                      # allocates the two buffers
        callback(algorithm)
        with count_allocated_elements() as tally:
            callback(algorithm)
        self.assertEqual(0, tally['containers'])


class TestInnerSolverBreakdown(CCPiTestClass):
    """
    An inner solve that converges before its iteration budget must stop, not
    produce nan. IRLS runs the inner solver for a fixed count every outer
    iteration, so a single breakdown poisons the whole reconstruction.
    """

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.data = self.ig.allocate('random', seed=41)

    def degenerate_solver(self):
        r"""
        :math:`K = [I; \alpha I]` gives :math:`K^T K = (1 + \alpha^2) I`, a
        multiple of the identity, which LSQR solves exactly in one iteration.
        Everything after that is a breakdown of the bidiagonalisation.
        """
        return LSQR(operator=IdentityOperator(self.ig), data=self.data,
                    initial=self.ig.allocate(0), alpha=0.7, weighted=True)

    def test_lsqr_stops_instead_of_returning_nan(self):
        solver = self.degenerate_solver()
        solver.run(10, verbose=0)
        self.assertTrue(numpy.isfinite(solver.get_output().as_array()).all())
        self.assertTrue(numpy.isfinite(solver.loss).all())

    def test_the_exact_solution_survives_the_extra_iterations(self):
        """Stopping keeps the converged iterate rather than overwriting it."""
        one_step = self.degenerate_solver()
        one_step.run(1, verbose=0)
        many = self.degenerate_solver()
        many.run(10, verbose=0)
        self.assertNumpyArrayAlmostEqual(one_step.get_output().as_array(),
                                         many.get_output().as_array(),
                                         decimal=6)

    def test_irls_survives_a_degenerate_inner_solve(self):
        irls = IRLS(inner_solver=self.degenerate_solver(),
                    max_inner_iteration=10)
        irls.run(4, verbose=0)
        self.assertTrue(numpy.isfinite(irls.get_output().as_array()).all())

    def test_a_nondegenerate_problem_still_uses_its_whole_budget(self):
        """The guard must not fire early on an ordinary problem."""
        operator, data = small_least_squares(rows=12, columns=6, seed=3)
        solver = LSQR(operator=operator, data=data,
                      initial=operator.domain_geometry().allocate(0),
                      alpha=0.3, weighted=True)
        solver.run(5, verbose=0)
        self.assertEqual(5, solver.iteration)


class TestIRLSStoppingIntegration(CCPiTestClass):

    def setUp(self):
        self.operator, self.data = small_least_squares(rows=12, columns=6,
                                                       seed=3)
        self.geometry = self.operator.domain_geometry()

    def irls(self, tol=None):
        inner = LSQR(operator=self.operator, data=self.data,
                     initial=self.geometry.allocate(0), alpha=0.3,
                     weighted=True)
        return IRLS(inner_solver=inner, max_inner_iteration=10, tol=tol)

    def test_tol_is_off_by_default(self):
        irls = self.irls()
        self.assertIsNone(irls.tol)
        recorder = RecordingCallback()
        irls.run(6, callbacks=[recorder], verbose=0)
        self.assertEqual(6, irls.iteration)

    def test_tol_stops_the_outer_loop_early(self):
        irls = self.irls(tol=1e-3)
        irls.run(40, verbose=0)
        self.assertLess(irls.iteration, 40)

    def test_stopping_early_reaches_the_same_answer(self):
        stopped = self.irls(tol=1e-6)
        stopped.run(40, verbose=0)
        full = self.irls()
        full.run(40, verbose=0)
        self.assertNumpyArrayAlmostEqual(full.get_output().as_array(),
                                         stopped.get_output().as_array(),
                                         decimal=4)

    def test_a_caller_supplied_stopper_is_not_duplicated(self):
        irls = self.irls(tol=1e-3)
        mine = IRLSEarlyStopping(epsilon=1e-12, verbose=0)
        irls.run(5, callbacks=[mine], verbose=0)
        # With epsilon=1e-12 the caller's stopper never fires, so if IRLS had
        # also attached its own at 1e-3 the loop would have stopped short.
        self.assertEqual(5, irls.iteration)

    def test_a_tighter_tolerance_runs_longer(self):
        loose = self.irls(tol=1e-2)
        loose.run(40, verbose=0)
        tight = self.irls(tol=1e-8)
        tight.run(40, verbose=0)
        self.assertLess(loose.iteration, tight.iteration)


if __name__ == '__main__':
    unittest.main()
