#  Copyright 2026 United Kingdom Research and Innovation
#  Copyright 2026 The University of Manchester
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
# CIL Developers, listed at:
# https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt


import unittest

import numpy

from cil.framework import ImageGeometry, VectorData
from cil.optimisation.algorithms import CGLS, IRLS, LSQR
from cil.optimisation.operators import (DiagonalOperator, GradientOperator,
                                        IdentityOperator, MatrixOperator,
                                        TikhonovOperator, WaveletOperator)
from cil.optimisation.utilities.hybrid import (HybridRule, DiscrepHybridRule,
                                               GCVHybridRule)
from cil.optimisation.utilities.hybrid.HybridRule import (
    KrylovState, build_bidiagonal, find_optimal_alpha_via_grid,
    projected_residual_norm_sq)

from testclass import CCPiTestClass
from utils import initialise_tests

initialise_tests()


class RecorderRule(HybridRule):
    # records the bidiagonalisation and never selects

    def _calculate_optimal_regalpha(self, state, upper_bound):
        return numpy.nan

    def evaluate_objective(self, regalpha, state):
        return 0.0


class QuadraticLogRule(HybridRule):

    def evaluate_objective(self, regalpha, state):
        return (numpy.log10(regalpha) + 4.0) ** 2


def ill_posed_problem(m=60, n=40, noise=1e-2, seed=7, decay=None):
    # geometric singular values by default, k^-decay otherwise
    rng = numpy.random.RandomState(seed)
    U, _ = numpy.linalg.qr(rng.randn(m, m))
    V, _ = numpy.linalg.qr(rng.randn(n, n))
    if decay is None:
        s = numpy.logspace(0, -3, n)
    else:
        s = (1.0 + numpy.arange(n)) ** -decay
    A = (U[:, :n] * s) @ V.T
    x_true = V @ (s * rng.randn(n))
    e = rng.randn(m)
    e *= noise * numpy.linalg.norm(A @ x_true) / numpy.linalg.norm(e)
    b = A @ x_true + e
    return A, b, x_true, numpy.linalg.norm(e)


def build_lsqr(A, b, alpha=0, **kwargs):
    operator = MatrixOperator(A.astype(numpy.float64))
    data = VectorData(numpy.asarray(b, dtype=numpy.float64))
    return LSQR(operator=operator, data=data, alpha=alpha, **kwargs)


def run_rules(A, b, rules, iterations=35):
    for rule in rules.values():
        build_lsqr(A, b, alpha=rule).run(iterations, verbose=0)
    return rules


def oracle_alpha(A, b, x_true):
    # the alpha minimising the true error of the dense Tikhonov solution
    U, s, Vt = numpy.linalg.svd(A, full_matrices=False)
    coefficients = U.T @ b
    grid = numpy.geomspace(1e-6, 10, 4000)
    errors = [numpy.linalg.norm(
        Vt.T @ (s / (s ** 2 + alpha ** 2) * coefficients) - x_true)
        for alpha in grid]
    return float(grid[int(numpy.argmin(errors))])


class TestProjectionConvention(CCPiTestClass):

    ITERATIONS = 8

    def setUp(self):
        self.A, self.b, _, _ = ill_posed_problem()
        self.rule = RecorderRule()
        self.lsqr = build_lsqr(self.A, self.b, alpha=self.rule)
        self.lsqr.run(self.ITERATIONS, verbose=0)

    def test_projected_matrix_shape_is_one_more_row_than_columns(self):
        Bk = build_bidiagonal(self.rule.history.gkb_alphas,
                              self.rule.history.gkb_betas)
        self.assertEqual((self.ITERATIONS + 1, self.ITERATIONS), Bk.shape)

    def test_projected_residual_matches_the_true_residual(self):
        state = self.rule.return_krylov_state()
        projected = projected_residual_norm_sq(0.0, state, self.rule.b_norm)
        x = self.lsqr.get_output().as_array()
        true = float(numpy.sum(numpy.square(self.A @ x - self.b)))
        self.assertAlmostEqual(true, projected, delta=1e-3 * true)


class TestProjectedTikhonovOracle(CCPiTestClass):

    def setUp(self):
        rng = numpy.random.RandomState(3)
        self.B = numpy.zeros((8, 7))
        self.B[numpy.arange(7), numpy.arange(7)] = rng.rand(7) + 0.5
        self.B[numpy.arange(1, 8), numpy.arange(7)] = rng.rand(7) + 0.5
        self.b_norm = 2.5
        self.state = KrylovState(bk=self.B)

    def test_projected_residual_norm(self):
        rhs = numpy.zeros(15)
        rhs[0] = self.b_norm
        for regalpha in (1e-3, 0.1, 1.0):
            with self.subTest(regalpha=regalpha):
                stacked = numpy.vstack([self.B, regalpha * numpy.eye(7)])
                y = numpy.linalg.lstsq(stacked, rhs, rcond=None)[0]
                expected = float(numpy.sum(numpy.square(self.B @ y - rhs[:8])))
                self.assertAlmostEqual(
                    expected,
                    projected_residual_norm_sq(regalpha, self.state,
                                               self.b_norm),
                    places=10)


class TestGridSearchRobustness(CCPiTestClass):

    BOUNDS = (1e-12, 1e2)

    def search(self, func, **kwargs):
        return find_optimal_alpha_via_grid(func, bounds=self.BOUNDS, **kwargs)

    def test_refinement_reaches_optimiser_precision(self):
        func = lambda a: (numpy.log10(a) + 4.0) ** 2
        coarse = self.search(func, refine=False)
        refined = self.search(func)
        self.assertLess(abs(refined.best_alpha - 1e-4) / 1e-4, 1e-6)
        self.assertIn(coarse.best_alpha, coarse.alpha_grid)

    def test_finds_the_global_minimum(self):
        dip = lambda c, w, d: lambda a: -d * numpy.exp(
            -((numpy.log10(a) - c) ** 2) / w)
        multimodal = lambda a: dip(-8, 0.05, 0.5)(a) + dip(1, 0.05, 1.0)(a)
        nan_below = lambda a: (numpy.nan if a < 1e-6
                               else (numpy.log10(a) + 2) ** 2)
        for func, expected in ((multimodal, 1.0), (nan_below, -2.0)):
            result = self.search(func)
            self.assertAlmostEqual(expected, numpy.log10(result.best_alpha),
                                   places=3)

    def test_an_all_nan_objective_reports_nan_rather_than_raising(self):
        result = self.search(lambda a: numpy.nan)
        self.assertTrue(numpy.isnan(result.best_alpha))


class TestSolveAt(CCPiTestClass):

    def setUp(self):
        self.A, self.b, _, _ = ill_posed_problem()

    def explored(self, iterations=6):
        rule = RecorderRule()
        lsqr = build_lsqr(self.A, self.b, alpha=rule)
        lsqr.run(iterations, verbose=0)
        return lsqr, rule

    def test_matches_a_fresh_fixed_alpha_run(self):
        explorer, rule = self.explored(6)
        x_before = explorer.x
        rule.selected_alpha = 0.3
        second_pass = explorer.solve_at()
        self.assertIs(x_before, explorer.x)
        self.assertTrue(explorer._realised)
        self.assertIsNone(explorer.rebuild_solver)

        reference = build_lsqr(self.A, self.b, alpha=0.3, form='standard')
        reference.run(6, verbose=0)
        self.assertNumpyArrayAlmostEqual(
            reference.get_output().as_array(),
            second_pass.as_array(), decimal=6)

    def test_a_nonzero_initial_realises_in_block_form(self):
        operator = MatrixOperator(self.A)
        data = VectorData(numpy.asarray(self.b, dtype=numpy.float64))
        init = operator.domain_geometry().allocate(1.0)
        rule = RecorderRule()
        lsqr = LSQR(initial=init, operator=operator, data=data, alpha=rule)
        lsqr.run(6, verbose=0)
        rule.selected_alpha = 0.3
        solution = lsqr.solve_at()
        self.assertTrue(lsqr._realised)
        self.assertIsNotNone(lsqr.rebuild_solver)
        self.assertTrue(lsqr.rebuild_solver.block_form)
        self.assertIs(lsqr.operator.struct_operator,
                      lsqr.rebuild_solver.operator.struct_operator)

        reference = LSQR(initial=init, operator=operator, data=data,
                         alpha=0.3, form='block')
        reference.run(6, verbose=0)
        numpy.testing.assert_allclose(solution.as_array(),
                                      reference.get_output().as_array(),
                                      rtol=1e-6, atol=1e-10)
        numpy.testing.assert_allclose(solution.as_array(),
                                      lsqr.get_output().as_array())

    def test_survives_early_breakdown(self):
        rng = numpy.random.RandomState(0)
        A = numpy.eye(5) + 0.01 * rng.rand(5, 5)
        rule = RecorderRule()
        lsqr = build_lsqr(A, rng.rand(5), alpha=rule)
        lsqr.run(50, verbose=0)
        rule.selected_alpha = 1e-6
        solution = lsqr.solve_at()
        self.assertTrue(numpy.isfinite(solution.as_array()).all())

    def test_raises(self):
        lsqr = build_lsqr(self.A, self.b, alpha=0.3)
        lsqr.run(4, verbose=0)
        with self.assertRaisesRegex(ValueError, "hybrid"):
            lsqr.solve_at()
        lsqr, _ = self.explored(4)
        with self.assertRaisesRegex(ValueError, "not selected"):
            lsqr.solve_at()


class TestRuleLifecycle(CCPiTestClass):

    def setUp(self):
        self.A, self.b, _, self.noise_norm = ill_posed_problem()

    def test_first_call_seeds_without_selecting(self):
        rule = RecorderRule()
        lsqr = build_lsqr(self.A, self.b, alpha=rule)
        lsqr.run(1, verbose=0)
        # seed pair plus one update pair, one selection attempt
        self.assertEqual(2, len(rule.history.gkb_alphas))
        self.assertEqual(1, len(rule.history.regalphas))
        self.assertEqual(60, rule.m)
        self.assertEqual(40, rule.n)

    def test_a_rule_is_just_its_objective(self):
        rule = QuadraticLogRule()
        rule.seed(initial_alpha=1.0, initial_beta=2.0)
        rule.update(0.8, 0.6)
        self.assertAlmostEqual(-4.0, numpy.log10(rule.selected_alpha),
                               places=5)

    def test_saturation_stops_the_pass_unless_select_only(self):
        for mode in ('stop', 'select_only'):
            with self.subTest(mode=mode):
                rule = GCVHybridRule(tol=1e30, mode=mode)
                lsqr = build_lsqr(self.A, self.b, alpha=rule)
                lsqr.run(30, verbose=0)
                self.assertTrue(rule.converged)
                if mode == 'stop':
                    self.assertLess(lsqr.iteration, 30)
                else:
                    self.assertEqual(30, lsqr.iteration)
                    self.assertEqual(30, lsqr._pass_iterations)
                    self.assertEqual(31, len(rule.history.gkb_alphas))
                    self.assertTrue(numpy.isfinite(rule.selected_alpha))

    def test_match_mode_realises_over_the_pass_length(self):
        rule = DiscrepHybridRule(noise_level_estimate=self.noise_norm,
                                 mode='match')
        lsqr = build_lsqr(self.A, self.b, alpha=rule)
        lsqr.run(40, verbose=0)
        pass_length = len(rule.history.gkb_alphas) - 1
        self.assertLess(pass_length, 40)
        lsqr.solve_at()
        self.assertEqual(pass_length, lsqr._pass_iterations)

    def test_reinitialising_the_solver_re_arms_without_duplicating(self):
        rule = RecorderRule()
        lsqr = build_lsqr(self.A, self.b, alpha=rule)
        self.assertEqual(1, len(rule.history.gkb_alphas))
        lsqr.initialise_variables()
        self.assertEqual(1, len(rule.history.gkb_alphas))

    def test_seed_re_arms_and_archives_the_previous_pass(self):
        rule = RecorderRule()
        rule.seed(initial_alpha=1.0, initial_beta=2.0)
        self.assertEqual([], rule.pass_histories)
        rule.update(0.5, 0.4)
        first = rule.history
        rule.seed(initial_alpha=1.0, initial_beta=2.0)
        self.assertEqual(1, len(rule.history.gkb_alphas))
        self.assertFalse(rule.converged)
        self.assertTrue(numpy.isnan(rule.selected_alpha))
        self.assertEqual([first], rule.pass_histories)
        # an empty pass is not archived
        rule.seed(initial_alpha=1.0, initial_beta=2.0)
        self.assertEqual([first], rule.pass_histories)
        rule.update(0.5, 0.4)
        rule.seed(initial_alpha=1.0, initial_beta=2.0)
        self.assertEqual(2, len(rule.pass_histories))

    def test_irls_saturation_ends_the_outer_loop(self):
        rule = DiscrepHybridRule(noise_level_estimate=self.noise_norm,
                                 tol=0.1, mode='select_only')
        irls = IRLS(operator=MatrixOperator(self.A), data=VectorData(self.b),
                    alpha=rule, inner_solver=LSQR, max_inner_iteration=25)
        irls.run(20, verbose=0)
        self.assertGreaterEqual(len(irls.selected_alphas), 3)
        self.assertTrue(all(numpy.isfinite(irls.selected_alphas)))
        self.assertLess(irls.iteration, 20)
        newest = irls.selected_alphas[-1]
        for a in irls.selected_alphas[-3:]:
            self.assertLess(abs(a - newest) / abs(newest), 0.1)


class TestRuleCrossConsistency(CCPiTestClass):

    def test_selections_agree_with_the_oracle_and_each_other(self):
        for seed in (7, 23):
            with self.subTest(seed=seed):
                A, b, x_true, noise = ill_posed_problem(seed=seed)
                rules = run_rules(A, b, {
                    'discrep': DiscrepHybridRule(noise_level_estimate=noise),
                    'gcv': GCVHybridRule(adaptive_weight=False),
                    'wgcv': GCVHybridRule(adaptive_weight=True)})
                selected = {name: rule.selected_alpha
                            for name, rule in rules.items()}
                oracle = oracle_alpha(A, b, x_true)
                for value in selected.values():
                    self.assertLess(value / oracle, 10.0)
                    self.assertGreater(value / oracle, 0.1)
                self.assertLess(max(selected.values())
                                / min(selected.values()), 10.0)
                # given the exact noise level, discrepancy lands near the oracle
                self.assertLess(selected['discrep'] / oracle, 3.0)
                self.assertGreater(selected['discrep'] / oracle, 1 / 3.0)
                ratio = selected['wgcv'] / selected['gcv']
                self.assertGreater(ratio, 0.5)
                self.assertLess(ratio, 2.0)

    def test_no_over_regularisation_on_the_slow_decay_spectrum(self):
        for seed in (7, 23):
            with self.subTest(seed=seed):
                A, b, x_true, _ = ill_posed_problem(200, 150, seed=seed,
                                                    decay=0.7)
                rules = run_rules(A, b, {
                    'gcv': GCVHybridRule(adaptive_weight=False),
                    'wgcv': GCVHybridRule(adaptive_weight=True)})
                wgcv = rules['wgcv'].selected_alpha
                oracle = oracle_alpha(A, b, x_true)
                self.assertLess(wgcv / rules['gcv'].selected_alpha, 2.0)
                self.assertLess(wgcv / oracle, 2.0)
                self.assertGreater(wgcv / oracle, 0.3)


class TestGCVHybridRule(CCPiTestClass):

    def seeded_rule(self, regalphas):
        rule = GCVHybridRule(adaptive_weight=False, tol=1e-3)
        rule.seed(initial_alpha=1.0, initial_beta=1.0, data_size=60,
                  domain_size=40)
        rule.history.regalphas = list(regalphas)
        rule.selected_alpha = rule.history.regalphas[-1]
        rule.gcv_history.Ghats = [1.0] * 12 + [1.1, 1.2]
        return rule

    def test_convergence_waits_for_the_window_and_the_pick(self):
        # a repeated pick below a still-falling sigma_min is provisional
        rule = self.seeded_rule([2.7e3] * 14)
        rule._last_upper_bound = 5e5
        rule._window_lows = [4.5e3 * 0.98 ** k for k in range(14)]
        self.assertFalse(rule._check_convergence())
        rule._window_lows = [4.5e3] * 14
        self.assertTrue(rule._check_convergence())

        # rising Ghats stop only once both the window and the pick are still
        rule = self.seeded_rule(numpy.geomspace(1e-1, 1e-3, 14))
        rule._last_upper_bound = 1.0
        rule._window_lows = [0.5 ** k for k in range(1, 15)]
        self.assertFalse(rule._check_convergence())
        rule._window_lows = [0.5] * 14
        self.assertFalse(rule._check_convergence())
        rule.history.regalphas = [1e-3] * 14
        rule.selected_alpha = 1e-3
        self.assertTrue(rule._check_convergence())

    def test_a_floor_parked_minimum_is_no_selection(self):
        rule = GCVHybridRule(adaptive_weight=False)
        rule.seed(initial_alpha=1.0, initial_beta=1.0, data_size=60,
                  domain_size=40)
        rule.evaluate_objective = lambda regalpha, state: numpy.log10(regalpha)
        rule.update(0.8, 0.6)
        self.assertTrue(numpy.isnan(rule.selected_alpha))

    def test_omegas_are_clamped_and_averaged(self):
        A, b, _, _ = ill_posed_problem()
        awgcv = run_rules(A, b, {
            'wgcv': GCVHybridRule(adaptive_weight=True)})['wgcv']
        omegas = numpy.asarray(awgcv.gcv_history.omegas)
        self.assertGreater(len(omegas), 0)
        self.assertTrue((omegas <= 1.0).all())
        self.assertTrue((omegas >= 0.0).all())
        self.assertAlmostEqual(float(numpy.mean(omegas)), awgcv.omega,
                               places=12)


class TestHybridFormPlumbing(CCPiTestClass):

    def setUp(self):
        self.ig = ImageGeometry(8, 8)
        self.haar = WaveletOperator(self.ig, wname='haar',
                                    bnd_cond='periodization')
        # decaying weights, so that K = A L^* is ill-posed
        rng = numpy.random.RandomState(11)
        weights = self.ig.allocate(0)
        weights.fill(numpy.geomspace(1.0, 1e-2, 64).reshape(8, 8)
                     .astype(numpy.float32))
        self.A = DiagonalOperator(weights)
        clean = self.ig.allocate(0)
        clean.fill(rng.rand(8, 8).astype(numpy.float32))
        self.data = self.A.direct(clean)
        self.data += 0.02 * float(self.data.norm()) / 8 * self.ig.allocate(1.0)

    def test_a_rule_as_alpha_forces_the_standard_form(self):
        lsqr = LSQR(operator=self.A, data=self.data, struct_operator=self.haar,
                    alpha=GCVHybridRule(), form='block')
        self.assertTrue(lsqr.hybrid)
        self.assertTrue(lsqr.standard_form)
        self.assertFalse(lsqr.block_form)
        self.assertIsInstance(lsqr.operator, TikhonovOperator)

    def test_a_non_invertible_struct_operator_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"Forming L\^-1"):
            LSQR(operator=self.A, data=self.data,
                 struct_operator=GradientOperator(self.ig, method='backward'),
                 alpha=GCVHybridRule())

    def test_the_operator_wraps_at_alpha_zero(self):
        A = IdentityOperator(self.ig)
        data = self.ig.allocate(1.0)
        bare = LSQR(operator=A, data=data, struct_operator=self.haar,
                    alpha=0.0)
        hybridised = LSQR(operator=A, data=data, struct_operator=self.haar,
                          alpha=RecorderRule())
        self.assertIs(A, bare.operator)
        self.assertIsInstance(hybridised.operator, TikhonovOperator)
        self.assertEqual(0.0, hybridised.operator.regalpha)
        self.assertEqual(0, hybridised.regalpha)

    def test_the_selection_pass_starts_from_zero(self):
        A, b, _, _ = ill_posed_problem()
        for operator, data, struct in (
                (self.A, self.data, self.haar),
                (MatrixOperator(A), VectorData(b), None)):
            rule = GCVHybridRule(mode='select_only')
            lsqr = LSQR(initial=operator.domain_geometry().allocate(1.0),
                        operator=operator, data=data,
                        struct_operator=struct, alpha=rule)
            lsqr.run(10, verbose=0)
            # seeded from zero: beta_1 = ||b||
            self.assertAlmostEqual(rule.history.gkb_betas[0]
                                   / data.norm(), 1.0, places=5)

    def test_the_selection_pass_starts_from_zero_with_a_block_range(self):
        # beta_1 = ||P b||, the data deflated along the constants
        rule = GCVHybridRule(mode='select_only')
        irls = IRLS(operator=self.A, data=self.data, alpha=rule,
                    struct_operator=GradientOperator(self.ig),
                    max_inner_iteration=10)
        irls.run(3, verbose=0)
        deflated = irls.inner_solver.operator._deflate(self.data).norm()
        for history in rule.pass_histories + [rule.history]:
            self.assertAlmostEqual(history.gkb_betas[0] / deflated, 1.0,
                                   places=5)

    def test_hybrid_get_output_realises_the_selection(self):
        rule = GCVHybridRule()
        lsqr = LSQR(operator=self.A, data=self.data, struct_operator=self.haar,
                    alpha=rule)
        self.assertTrue(lsqr.standard_form)
        lsqr.run(30, verbose=0)
        exploratory = lsqr.operator.map_to_solution(lsqr.x).as_array().copy()

        u = lsqr.get_output().as_array().copy()
        self.assertTrue(lsqr._realised)
        self.assertIsNone(lsqr.rebuild_solver)
        self.assertEqual(lsqr.regalpha, rule.selected_alpha)
        self.assertFalse(numpy.allclose(u, exploratory))

        # a second call returns the same realised solution
        reference = LSQR(operator=self.A, data=self.data,
                         struct_operator=self.haar,
                         alpha=rule.selected_alpha, form='standard')
        reference.run(lsqr._pass_iterations, verbose=0)
        numpy.testing.assert_allclose(u, lsqr.get_output().as_array(),
                                      rtol=1e-6)
        numpy.testing.assert_allclose(u, reference.get_output().as_array(),
                                      rtol=1e-4)

    def test_irls_runs_the_two_pass_hybrid_loop(self):
        rule = GCVHybridRule()
        irls = IRLS(operator=self.A, data=self.data, alpha=rule,
                    struct_operator=self.haar, inner_solver=LSQR,
                    max_inner_iteration=30)
        self.assertTrue(irls.hybrid)
        self.assertTrue(irls.warm_start)
        self.assertIs(irls.weighted_operator,
                      irls.inner_solver.operator.struct_operator)

        irls.run(3, verbose=0)

        self.assertEqual(3, len(irls.selected_alphas))
        self.assertTrue(all(numpy.isfinite(a) and a > 0
                            for a in irls.selected_alphas))
        self.assertEqual(rule.selected_alpha, irls.selected_alphas[-1])
        inner = irls.inner_solver
        self.assertTrue(inner._realised)
        self.assertEqual(irls.selected_alphas[-1], inner.regalpha)
        # pass 2 warm starts from the previous solution on the block rebuild
        self.assertGreater(float(inner.initial.norm()), 0)
        self.assertIs(irls.weighted_operator,
                      inner.rebuild_solver.operator.struct_operator)
        numpy.testing.assert_allclose(
            irls.get_output().as_array(),
            inner.operator.map_to_solution(inner.x).as_array())

    def test_irls_reuses_the_previous_selection_when_a_rule_declines(self):
        class FirstPassOnlyRule(HybridRule):
            def _calculate_optimal_regalpha(self, state, upper_bound):
                return 0.05 if not self.pass_histories else numpy.nan

            def evaluate_objective(self, regalpha, state):
                return 0.0

        rule = FirstPassOnlyRule()
        irls = IRLS(operator=self.A, data=self.data, alpha=rule,
                    struct_operator=self.haar, inner_solver=LSQR,
                    max_inner_iteration=8)
        irls.run(3, verbose=0)
        self.assertEqual([0.05, 0.05, 0.05], irls.selected_alphas)
        self.assertEqual(0.05, rule.selected_alpha)
        self.assertEqual(0.05, irls.inner_solver.regalpha)

    def test_irls_select_only_runs_full_passes_until_saturation(self):
        rule = GCVHybridRule(tol=1e30, mode='select_only')
        irls = IRLS(operator=self.A, data=self.data, alpha=rule,
                    struct_operator=self.haar, inner_solver=LSQR,
                    max_inner_iteration=12)
        irls.run(6, verbose=0)
        # the last three selections agree, ending the run inside its budget
        self.assertLess(irls.iteration, 6)
        self.assertEqual(3, len(irls.selected_alphas))
        self.assertTrue(all(numpy.isfinite(a) and a > 0
                            for a in irls.selected_alphas))
        self.assertEqual(13, len(rule.pass_histories[0].gkb_alphas))
        self.assertEqual(13, len(rule.history.gkb_alphas))
        self.assertEqual(rule.selected_alpha, irls.selected_alphas[-1])

    def test_raises(self):
        with self.assertRaisesRegex(ValueError, "solve_at"):
            IRLS(operator=self.A, data=self.data, alpha=GCVHybridRule(),
                 struct_operator=self.haar, inner_solver=CGLS,
                 max_inner_iteration=8)
        irls = IRLS(operator=self.A, data=self.data, alpha=RecorderRule(),
                    struct_operator=self.haar, inner_solver=LSQR,
                    max_inner_iteration=8)
        with self.assertRaisesRegex(RuntimeError, "no selection"):
            irls.run(1, verbose=0)


class TestRuleConstructors(CCPiTestClass):

    def test_raises(self):
        with self.assertRaises(ValueError):
            DiscrepHybridRule(noise_level_estimate=0.0)
        with self.assertRaises(ValueError):
            GCVHybridRule(mode='sometimes')
        for cls in (GCVHybridRule, RecorderRule):
            with self.assertRaises(ValueError):
                cls(tol=0)


class TestRulesEndToEnd(CCPiTestClass):

    def test_discrepancy(self):
        A, b, x_true, noise_norm = ill_posed_problem()
        rule = DiscrepHybridRule(noise_level_estimate=noise_norm)
        lsqr = build_lsqr(A, b, alpha=rule)
        lsqr.run(25, verbose=0)
        self.assertTrue(numpy.isfinite(rule.selected_alpha))
        self.assertGreater(rule.selected_alpha, 0)

        # the second-pass residual sits at the noise level
        x = lsqr.solve_at().as_array()
        residual = numpy.linalg.norm(A @ x - b)
        self.assertLess(residual, 3 * noise_norm)
        self.assertGreater(residual, noise_norm / 3)

        plain = build_lsqr(A, b)
        plain.run(25, verbose=0)
        error = lambda u: numpy.linalg.norm(u - x_true)
        self.assertLess(error(x), error(plain.get_output().as_array()))


if __name__ == '__main__':
    unittest.main()
