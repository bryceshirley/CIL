"""
Unit tests for Hybrid LSQR Regularization Parameter Selection Rules.

1. Base Class Infrastructure Verification:
   We test the `UpdateRegBase` logic (inherited by all rules) to ensure:
   - Proper SVD decomposition of the projected bidiagonal matrix B_k.
   - Correct extraction of the first row of the left singular vectors (u1).
   - Correct tracking of the regularization parameter history (regalphavec).
   - Functional convergence/saturation logic based on relative changes in alpha.

2. Mocking Bidiagonal Systems: 
   Synthetic (k+1 x k) bidiagonal matrices are generated with controlled 
   singular value decay to simulate ill-posed systems. This allows for 
   fast, deterministic testing without a full forward/adjoint operator.
   - Controlled right-hand sides (b0) are used to simulate noise levels.
   - Known optimal regularization parameters (alpha_opt) are precomputed 
     for validation?
   - Check alpha updates against expected values for each rule.

3. Rule-Specific Mathematical Verification:
    - GCV/WGCV: Checks minimization of the GCV functional and adaptive 
      weighting (omega) calculations.
    - Discrepancy Principle: Checks the root-finding logic identifies 
      the alpha that satisfies the Morozov discrepancy principle (residual 
      norm approx. noise level).
    - L-Curve: Verifies the exact analytical 1st and 2nd derivatives used 
      to find the point of maximum curvature (corner). And ensures curvature
      calculations are correct.
"""
import unittest
import numpy as np
import scipy
from cil.optimisation.utilities.HybridUpdateReg import (
    UpdateRegBase, UpdateRegDiscrep,
    UpdateRegLcurve, UpdateRegGCV
)

class RegRuleInfrastructureTestsMixin:
    """
    Mixin providing common infrastructure tests. 
    Expects inheriting classes to run 
    setup_bidiagonal_system method in their setUp(),
    for a given RuleClass.
    """
    def setup_defaults(self,RuleClass):
        """
        Initialize default parameters for tests.

        Parameters
        ----------
        RuleClass : class
            The regularization parameter selection rule class to be tested.
        """
        self.m = 100
        self.n = 100
        self.tol = 1e-3
        self.b_norm = 1.0
        self.RuleClass = RuleClass

    def setup_noisy_system(self, rule=None, noise_level = 0.1):
        """
        Factory to create a rule instance with an injected ill-conditioned state.
        Bypasses SVD computation to focus on rule-specific logic.
        """
        if rule is None:
            rule = self.RuleClass(self.tol, self.m, self.n)
        rule.iteration = 50
        # Ill-conditioned spectrum (10^1 to 10^-6)
        rule.Sb = np.logspace(1, -6, rule.iteration)
        rule.Sbsq = np.square(rule.Sb)
        rule.Sbmax = rule.Sb[0]
        rule.Sbmin = rule.Sb[-1]
        
        # Optimization Upper Bound
        rule.regalpha_high = rule.Sbmax * 1e3

        # Construct Projected RHS: Signal Decay (Picard Condition) + Flat Noise Tail
        # Add noise floor: these components are amplified by small singular values, 
        # creating the vertical 'noise' arm of the L-curve.
        # This is the discrete Picard condition violation.
        u1_raw = np.logspace(0, -3, rule.iteration + 1)
        u1_raw[-10:] = noise_level 
        u1 = u1_raw / np.linalg.norm(u1_raw) # Ensure ||u1|| = 1
        rule.u1_tail = u1[-1]
        rule.u1 = u1[:-1]

        # Construct Projected RHS: Signal Decay (Picard Condition) + Flat Noise Tail
        rule.b_norm = self.b_norm

        return rule

    def setup_small_projected_matrix(self):
        """
        Factory to create a small projected matrix.
        """
        # Small test matrix where we know the SVD result
        Bk = np.array([[10, 4, 0], 
                    [0, 1, 5], 
                    [2, 1, 3],
                    [2, 1, 3]]) # (k+1 x k)
        
        return Bk

    def setup_small_system_single_iteration(self,rule=None):
        """
        Factory to create a rule instance after a single iteration.
        """
        if rule is None:
            rule = self.RuleClass(self.tol, self.m, self.n)
        Bk = self.setup_small_projected_matrix()
        
        rule.update_regularizationparam(Bk, self.b_norm)
        return rule

    def test_initialize_subspace_components(self):
        '''Verify SVD components are correctly computed and stored.'''
        rule = self.RuleClass(self.tol, self.m, self.n)
        Bk = self.setup_small_projected_matrix()
        
        rule._initialize_subspace_components(Bk, self.b_norm)

        # Reference SVD
        U, s, _ = scipy.linalg.svd(Bk)
        
        # Verify singular values and projection vector
        np.testing.assert_array_almost_equal(rule.Sb, s)
        np.testing.assert_array_almost_equal(rule.Sbsq, s**2)
        np.testing.assert_array_almost_equal(rule.u1_tail, U[0, -1])
        np.testing.assert_array_almost_equal(rule.u1, U[0, :-1])

        # Verify bounds used for optimization
        self.assertEqual(rule.Sbmax, s[0])
        self.assertEqual(rule.Sbmin, s[-1])
        self.assertEqual(rule.regalpha_high, s[0])

    def test_convergence_logic(self):
        '''Verify convergence logic based on relative change in regalpha.'''
        tol = 0.1
        rule = self.RuleClass(tol, self.m, self.n)

        # Condition : |a[-1] - a[-2]| / (|a[-1]| + EPS) < tol
        rule.converged = False
        scenarios = [
            ([1.0, 1.05], True, "Within tol (5% change)"),
            ([1.0, 1.25], False, "Outside tol (25% change)"),
            ([0.0, 0.1], False, "From zero to non-zero"), 
            ([0.1, 0.100000001], True, "Stabilized near zero"),
            ([1.0, 0.95], True, "Decreased within tol"),
            ([1.0, 1.0], True, "Static values should trigger convergence."),
            ([0.0, 0.0], True, "Zero values should handle EPS safely."),
        ]
        for values, expected, msg in scenarios:
            rule.regalpha_history = values
            rule.converged = False
            rule._run_convergence_checks()
            self.assertEqual(rule.converged, expected, msg)

    def test_print_status_formatting(self):
        '''Ensure status printing handles None regalpha gracefully.'''
        rule = self.RuleClass(self.tol, self.m, self.n)
        # Should not raise error
        rule._print_status()
    
    def test_filter_logics(self):
        """Combined test for residual and solution filter mathematics and limits."""
        rule = self.setup_small_system_single_iteration()
        alpha_val = 0.5
        large_alpha = rule.Sbmax * 1e6

        scenarios = [
            # (alpha, filter_type, expected_calc)
            (alpha_val, "res", (alpha_val**2) / (rule.Sbsq + alpha_val**2)),
            (0.0,       "res", np.zeros_like(rule.Sb)),
            (large_alpha, "res", np.ones_like(rule.Sb)),
            
            (alpha_val, "sol", rule.Sb / (rule.Sbsq + alpha_val**2)),
            (0.0,       "sol", 1.0 / rule.Sb),
            (large_alpha, "sol", np.zeros_like(rule.Sb))
        ]
        # Solution Filter: Sb / (Sb^2 + alpha^2)
        # Residual Filter: alpha^2 / (Sb^2 + alpha^2)

        for alpha, f_type, expected in scenarios:
            with self.subTest(alpha=alpha, f_type=f_type):
                calc = rule._residual_filter(alpha) if f_type == "res" else rule._solution_filter(alpha)
                np.testing.assert_array_almost_equal(calc, expected, decimal=10)

    def test_projected_norms_mathematics(self):
        """Combined test for solution and residual norm assembly and subspace handling."""
        # Perform iteration to initialize SVD components
        rule = self.setup_small_system_single_iteration()

        alpha = 0.5
        
        # Test 1: Residual Norm Assembly
        # Logic: \|b\|^2 * (\sum_{i=1}^k((filt * u_{i})^2) + u_{k+1}^2)
        res_filt = rule._residual_filter(alpha)
        b_norm_sq = rule.b_norm **2
        expected_res = b_norm_sq * (np.sum((res_filt * rule.u1)**2) + rule.u1_tail**2)
        self.assertAlmostEqual(rule._projected_residual_norm_sq(alpha), expected_res, places=12)

        # Test 2: Solution Norm Assembly
        # Logic: \|b\|^2 * \sum_{i=1}^k((filt * u_{i})^2)
        sol_filt = rule._solution_filter(alpha)
        expected_sol = b_norm_sq * np.sum((sol_filt * rule.u1)**2)
        self.assertAlmostEqual(rule._projected_solution_norm_sq(alpha), expected_sol, places=12)

    def test_norm_asymptotic_limits(self):
        """Verify norms at extreme alpha values."""
        rule = self.setup_small_system_single_iteration()
        large_alpha = rule.Sbmax * 1e6

        # Alpha -> Infinity Limits
        # Solution Norm: \|b\|^2 * \sum_{i=1}^k((filt * u_{i})^2)
        # Solution Filter: Sb / (Sb^2 + alpha^2) -> 0
        # Solution -> 0
        # Residual Norm: \|b\|^2 * (\sum_{i=1}^k((filt * u_{i})^2) + u_{k+1}^2)
        # Residual Filter: alpha^2 / (Sb^2 + alpha^2) -> 1
        # Residual -> \|b\|^2 * ((\sum_{i=1}^k((u_{i})^2) + u_{k+1}^2) = \|b\|^2
        self.assertAlmostEqual(rule._projected_residual_norm_sq(large_alpha), rule.b_norm**2, places=5)
        self.assertAlmostEqual(rule._projected_solution_norm_sq(large_alpha), 0.0, places=12)

        # Alpha -> 0 Limits (Solution only, as residual depends on Sbmin)
        # Solution Filter: Sb / (Sb^2 + alpha^2) -> 1 / sigma
        # Solution -> \|b\|^2 * \sum_{i=1}^k((u_{i}/\sigma_i)^2)
        # Residual Norm: \|b\|^2 * (\sum_{i=1}^k((filt * u_{i})^2) + u_{k+1}^2)
        # Residual Filter: alpha^2 / (Sb^2 + alpha^2) -> 0
        # Residual -> \|b\|^2 * (0 + u_{k+1}^2) = \|b\|^2 * u_{k+1}^2
        expected_ls_sol = (rule.b_norm **2) * np.sum((rule.u1 / rule.Sb)**2)
        self.assertAlmostEqual(rule._projected_solution_norm_sq(0.0), expected_ls_sol, places=12)
    
    def test_invalid_initialization(self):
        '''Verify that __init__ rejects non-positive dimensions and tolerances.'''

        # Test negative/zero dimensions
        with self.assertRaises(ValueError):
            self.RuleClass(regalpha_saturation_tol=1e-3, m=0, n=100)
        with self.assertRaises(ValueError):
            self.RuleClass(regalpha_saturation_tol=1e-3, m=100, n=-5)
        
        # Test non-positive tolerance
        with self.assertRaises(ValueError):
            self.RuleClass(regalpha_saturation_tol=0, m=100, n=100)
        with self.assertRaises(ValueError):
            self.RuleClass(regalpha_saturation_tol=-1e-4, m=100, n=100)

    def test_invalid_update_inputs(self):
        '''Verify that update_regularizationparam rejects inconsistent matrix shapes and types.'''
        rule = self.RuleClass(self.tol, self.m, self.n)
        # Test non-ndarray Bk
        with self.assertRaises(TypeError):
            rule.update_regularizationparam(Bk=[[1, 0], [0, 1]], b_norm=1.0)
        # Test Bk that isn't (k+1, k)
        invalid_shape_Bk = np.eye(5) # (5, 5) instead of (6, 5)
        with self.assertRaises(ValueError):
            rule.update_regularizationparam(invalid_shape_Bk, self.b_norm)
        # Test Bk exceeding operator dimensions
        huge_Bk = np.zeros((self.m + 2, self.m + 1))
        with self.assertRaises(ValueError):
            rule.update_regularizationparam(huge_Bk, self.b_norm)
        # Test non-scalar b_norm
        with self.assertRaises(TypeError):
            Bk = self.setup_small_projected_matrix()
            rule.update_regularizationparam(Bk, b_norm=[1.0, 0.0])
    
    def test_interface_return_signature(self):
        '''Verify _compute_next_regalpha returns either a float or None.'''
        rule = self.setup_small_system_single_iteration()
        new_alpha = rule._compute_next_regalpha()
        # Check for float or None
        self.assertTrue(new_alpha is None or np.isscalar(new_alpha), "Single return value must be None or scalar.")

    
    def test_plot_history_error_handling(self):
        '''Verify ValueError is raised when trying to plot empty history.'''
        # Setup rule without any history
        rule = self.RuleClass(self.tol, self.m, self.n)

        # 1. Test completely empty history
        with self.assertRaisesRegex(ValueError, "No regularization parameter history"):
            rule.plot_history(show_objective=False)

        # 2. Test missing objective history specifically
        # Manually add to regalpha but keep func_history empty
        rule.regalpha_history = [0.1, 0.2]
        with self.assertRaisesRegex(ValueError, "No function history available"):
            rule.plot_history(show_objective=True)
    
    def test_geometric_grid_logic(self):
        """Verify the grid generation, zero-handling, and index finding."""
        rule = self.setup_small_system_single_iteration()
        # Set a known alpha to check index finding
        rule.regalpha = 1.0
        
        # Scenario 1: Standard valid range
        grid, funcs, idx = rule._geometric_grid(regalpha_limits=(0.1, 10.0), num_points=50)
        self.assertEqual(len(grid), 50)
        self.assertEqual(len(funcs), 50)
        # 1.0 is the 25th point in a geomspace from 0.1 to 10 with 50 points
        self.assertAlmostEqual(grid[idx], rule.regalpha, delta=0.1)

        # Scenario 2: Handle zero lower bound (Numerical Stability)
        # Should not raise ValueError: 'Geometric sequence cannot include zero'
        grid_zero, _, _ = rule._geometric_grid(regalpha_limits=(0.0, 1.0))
        self.assertGreater(grid_zero[0], 0.0)
        self.assertEqual(grid_zero[0], rule.EPS)

        # Scenario 3: Invalid limits (Robustness)
        with self.assertRaises(ValueError):
            # stop < start
            rule._geometric_grid(regalpha_limits=(10.0, 0.1))
        with self.assertRaises(ValueError):
            # stop == start
            rule._geometric_grid(regalpha_limits=(1.0, 1.0))

class TestUpdateRegBase(unittest.TestCase, RegRuleInfrastructureTestsMixin):
    """
    Unit tests for the Base Class of Hybrid LSQR Regularization Parameter Selection Rules.
    """
    def setUp(self):
        '''Set up synthetic bidiagonal system for testing.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return 0.1
            def func(self, regalpha): return 0.0
        self.setup_defaults(RuleClass=MockRule)
        self.Bk =self.setup_small_projected_matrix()

    def test_base_state_management(self):
        '''Verify history tracking, iteration counting, and length-safety gating.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self):
                # iteration is Bk.shape[1] = 3
                return 0.3 / self.iteration 
            def func(self, regalpha): return 0

        rule = MockRule(self.tol, self.m, self.n)
        
        # 1. Check Iteration and History Appending
        # Bk.shape[1] is 5, so alpha should be 0.5 / 5 = 0.1
        rule.update_regularizationparam(self.Bk, self.b_norm)
        iteration = self.Bk.shape[1]
        self.assertEqual(rule.iteration, iteration)
        self.assertEqual(len(rule.regalpha_history), 1)
        self.assertAlmostEqual(rule.regalpha_history[0], 0.1)

        # 2. Test safety gate for short history
        # With only one alpha, _run_convergence_checks should exit silently
        rule.converged = False
        rule._run_convergence_checks()
        self.assertFalse(rule.converged, "Should not converge with a history length of 1.")

        # 3. Test multiple appends
        rule.update_regularizationparam(self.Bk, self.b_norm)
        self.assertEqual(len(rule.regalpha_history), 2)
    
    def test_history_tracking(self):
        '''Verify regalpha_history and func_history correctly append via parent method.'''
        # We use a Mock here to control the returns exactly, 
        # independent of the specific rule's math.
        '''Verify regalpha_history and func_history correctly append via parent method.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): 
                # self.iteration is 3 based on your Bk shape (4, 3)
                return 0.1 * self.iteration
            def func(self, regalpha): return 99.0

        rule = MockRule(self.tol, self.m, self.n)
        
        # First call: k=3, alpha = 0.3
        rule.update_regularizationparam(self.Bk, self.b_norm)
        
        # Second call: alpha = 0.3
        rule.update_regularizationparam(self.Bk, self.b_norm)
        
        # 1. Check alpha history (Expected: 0.1 * 3 = 0.3)
        self.assertEqual(len(rule.regalpha_history), 2)
        np.testing.assert_array_almost_equal(rule.regalpha_history, [0.3, 0.3])
        
        # 2. Check function value history
        self.assertEqual(len(rule.func_history), 2)
        np.testing.assert_array_almost_equal(rule.func_history, [99.0, 99.0])
        
        # 3. Check current state matches last entry
        self.assertAlmostEqual(rule.regalpha, 0.3)

        class MockAlphaOnly(UpdateRegBase):
            def _compute_next_regalpha(self): return 0.1
            def func(self, regalpha): return None

        rule = MockAlphaOnly(self.tol, self.m, self.n)
        rule.update_regularizationparam(self.Bk, self.b_norm)
        
        self.assertEqual(len(rule.regalpha_history), 1)
        self.assertEqual(len(rule.func_history), 0, "func_history should stay empty if None is returned.")


class TestUpdateRegDiscrep(unittest.TestCase, RegRuleInfrastructureTestsMixin):
    def setUp(self):
        '''Set up synthetic bidiagonal system for testing.'''
        self.setup_defaults(RuleClass=UpdateRegDiscrep)
    
    def test_discrepancy_monotonicity(self):
        '''
        Verify that alpha is a monotonically increasing function of noise
        level estimate.
        '''
        estimated_noises = [0.1, 0.2, 0.3, 0.4]
        alphas = []
        
        for n in estimated_noises:
            rule = UpdateRegDiscrep(self.tol, self.m, self.n, noise_level_estimate=n)
            rule = self.setup_small_system_single_iteration(rule=rule)
            alphas.append(rule.regalpha)
            self.assertGreater(rule.regalpha, 0.0)
            
        # Check that alphas are strictly increasing
        for i in range(len(alphas) - 1):
            self.assertLess(alphas[i], alphas[i+1], 
                           f"Alpha should increase with noise. Failed at index {i}")
    
    def test_zero_noise_limit(self):
        '''Verify that if noise level is below reachable residual, alpha is 0.'''
        # Set noise to 0.0
        rule = UpdateRegDiscrep(self.tol, self.m, self.n, noise_level_estimate=0.0)
        rule = self.setup_small_system_single_iteration(rule=rule)
        
        # The logic should realize that the residual at alpha=0 
        # is already > 0.0, so it should return the lower bound.
        self.assertEqual(rule.regalpha, 0.0, 
                         "Should return alpha_low (0.0) when noise is unreachable.")
    
    def test_residual_at_root(self):
        '''Verify that the residual at the selected alpha matches the noise level
        estimate.'''
        noise_level_estimate = 0.3
        rule = UpdateRegDiscrep(self.tol, self.m, self.n, noise_level_estimate)
        rule = self.setup_small_system_single_iteration(rule=rule)
        
        # func(alpha) should be ~0, meaning residual^2 - noise^2 = 0
        self.assertAlmostEqual(rule.func(rule.regalpha), 0.0, places=6)
    
    def test_discrepancy_mathematical_recalculation(self):
        '''Verify that the identified alpha satisfies the Discrepancy Principle.'''
        noise_level_estimate = 0.25
        rule = UpdateRegDiscrep(self.tol, self.m, self.n, noise_level_estimate)
        rule = self.setup_small_system_single_iteration(rule=rule)
        
        alpha = rule.regalpha
        # Manual math check:
        filt = (alpha**2) / (rule.Sbsq + alpha**2)
        
        # rule.Sbsq is length k, rule.u1 is length k.
        residual_sq = np.sum(np.square(filt * (rule.b_norm * rule.u1))) + \
                      np.square(rule.b_norm * rule.u1_tail)
        
        self.assertAlmostEqual(np.sqrt(residual_sq), noise_level_estimate, places=5)
    
    def test_discrepancy_root_finding_accuracy(self):
        """
        Numerical verification that the rule identifies the precise root where
        the residual norm matches the target noise level.
        """
        # 1. Setup: Create an ill-conditioned system with a known noise floor
        # We use a high noise_level (0.5) to ensure a distinct Picard violation.
        rule = self.setup_noisy_system(noise_level=0.5)
        
        # 2. Define Reachable Target:
        # We set the estimate to twice the absolute floor (|beta_tail|) to 
        # guarantee a valid zero-crossing exists within the search range.
        floor_norm = np.abs(rule.b_norm * rule.u1_tail)
        rule.noise_level_estimate = floor_norm * 2
        
        # 3. Execution: Solve for the regularization parameter alpha
        # This invokes the internal root-finding (e.g., Brent's method).
        new_alpha = rule._compute_next_regalpha()
        rule.regalpha = new_alpha

        # 4. Debugging: Optional visual inspection of the root crossing
        # rule.plot_function(filepath="discrepancy_root_test.png") 
        
        # 5. Numerical Verification:
        # A: The discrepancy function value phi(alpha) must be zero.
        self.assertAlmostEqual(rule.func(rule.regalpha), 0.0, places=5)
        
        # B: The actual projected residual norm must match the target noise squared.
        expected_res_sq = rule.noise_level_estimate**2
        actual_res_sq = rule._projected_residual_norm_sq(rule.regalpha)
        self.assertAlmostEqual(actual_res_sq, expected_res_sq, places=5)

        # 5. Define Unreachable Target: 
        # We set the estimate to half the floor. Since the residual cannot 
        # physically go below the noise floor, phi(alpha) will always be positive.
        rule.noise_level_estimate = floor_norm * 0.5
        
        # In many implementations, if no root is found, the solver returns 
        # the boundary value (e.g., alpha_min) or None. 
        # Adjust this based on your specific solver's failure behavior.
        failed_alpha = rule._compute_next_regalpha()
        
        # Assertion: The function at this alpha should NOT be zero 
        # (it should still be positive because the residual > estimate).
        self.assertGreater(rule.func(failed_alpha), 0.0)

class TestUpdateRegLcurve(unittest.TestCase, RegRuleInfrastructureTestsMixin):
    """
    Unit tests for the L-Curve Regularization Parameter Selection Rule.
    
    Verified Behaviors:
    1. **Analytical Accuracy**: Comparison of exact 1st/2nd derivatives of 
       projected norms against finite difference approximations.
    2. **Optimization Integrity**: Validation that `scipy.optimize.minimize` 
       consistently finds the global maximum of the curvature function.
    3. **Asymptotic Limits**: Testing behavior at the limits of alpha 
       (alpha -> 0 and alpha -> infinity).
    4. **Numerical Stability**: Handling of log-space transformations when 
       residual or solution norms are near machine epsilon.
    """

    def setUp(self):
        '''Set up synthetic bidiagonal system for testing.'''
        self.setup_defaults(RuleClass=UpdateRegLcurve)

    def test_curvature_peak_finding(self):
        """
        Construct a projected problem with a clear L-curve and verify
        that the selected alpha corresponds to the peak curvature.
        """
        rule = self.setup_noisy_system(noise_level=0.001)

        # Compute next regularization parameter  and curvature
        new_alpha = rule._compute_next_regalpha()
        rule.regalpha = new_alpha
        new_curv = rule.func(new_alpha)

        # Verification using a dense grid search
        regalpha_grid, curvatures, _ = rule._geometric_grid(num_points=400)
        grid_max_idx = np.argmax(curvatures)
        grid_alpha = regalpha_grid[grid_max_idx]
        grid_max_curv = curvatures[grid_max_idx]

        # # Plot for debugging 
        # rule.plot_function(filepath="l_curve.png")  # L-curve & curvature

        # Assertions: selected alpha should match grid peak
        self.assertAlmostEqual(
            np.log10(new_alpha), np.log10(grid_alpha), places=1,
            msg=f"Optimized alpha ({new_alpha:.2e}) differs from grid ({grid_alpha:.2e})"
        )
        self.assertAlmostEqual(new_curv, grid_max_curv, places=2)


    def test_derivative_consistency(self):
        """Compare analytical derivatives against finite difference results."""
        rule = self.setup_noisy_system()
        alpha = 0.5
        h = 1e-5  # Step size for finite difference

        # Analytical results
        R2_p_a, R2_pp_a, X2_p_a, X2_pp_a = rule._projected_norm_derivatives(alpha)

        # Finite Difference for First Derivatives
        # (f(a+h) - f(a-h)) / 2h
        R2_p_fd = (rule._projected_residual_norm_sq(alpha + h) - 
                   rule._projected_residual_norm_sq(alpha - h)) / (2 * h)
        X2_p_fd = (rule._projected_solution_norm_sq(alpha + h) - 
                   rule._projected_solution_norm_sq(alpha - h)) / (2 * h)

        # Finite Difference for Second Derivatives
        # (f(a+h) - 2f(a) + f(a-h)) / h^2
        R2_pp_fd = (rule._projected_residual_norm_sq(alpha + h) - 
                    2 * rule._projected_residual_norm_sq(alpha) + 
                    rule._projected_residual_norm_sq(alpha - h)) / (h**2)
        X2_pp_fd = (rule._projected_solution_norm_sq(alpha + h) - 
                    2 * rule._projected_solution_norm_sq(alpha) + 
                    rule._projected_solution_norm_sq(alpha - h)) / (h**2)

        # Comparisons (Relative tolerances because second derivatives are sensitive)
        np.testing.assert_allclose(R2_p_a, R2_p_fd, rtol=1e-5, err_msg="R2 first derivative mismatch")
        np.testing.assert_allclose(X2_p_a, X2_p_fd, rtol=1e-5, err_msg="X2 first derivative mismatch")
        np.testing.assert_allclose(R2_pp_a, R2_pp_fd, rtol=1e-3, err_msg="R2 second derivative mismatch")
        np.testing.assert_allclose(X2_pp_a, X2_pp_fd, rtol=1e-3, err_msg="X2 second derivative mismatch")

    def test_log_space_stability(self):
        """Ensure no NaN/Inf results when norms are extremely small."""
        rule = self.setup_noisy_system()
        
        # Test 1: Alpha -> 0 (Smallest possible singular value neighborhood)
        tiny_alpha = 1e-20
        curvature_tiny = rule.func(tiny_alpha)
        self.assertTrue(np.isfinite(curvature_tiny), "Curvature calculation failed at near-zero alpha.")

        # Test 2: Alpha -> Infinity
        huge_alpha = 1e20
        curvature_huge = rule.func(huge_alpha)
        self.assertTrue(np.isfinite(curvature_huge), "Curvature calculation failed at near-infinite alpha.")

    def test_alpha_monotonicity_with_noise(self):
        """
        Verify that the selected alpha increases as the noise floor increases.
        """

        # Scenario 1: Low noise floor
        rule = self.setup_noisy_system(noise_level=0.001)
        alpha_low_noise = rule._compute_next_regalpha()

        # Scenario 2: High noise floor
        rule_high_noise = self.setup_noisy_system(noise_level=0.1)
        alpha_high_noise = rule_high_noise._compute_next_regalpha()
        # Assertions
        self.assertIsNotNone(alpha_low_noise)
        self.assertIsNotNone(alpha_high_noise)
        self.assertGreater(
            alpha_high_noise, alpha_low_noise, 
            f"Alpha did not increase with noise: {alpha_high_noise:.2e} <= {alpha_low_noise:.2e}"
        )

    def test_lcurve_stability_high_noise(self):
        """
        Test the rule's behavior under extreme noise conditions.
        """
        # Create a system where the "signal" is very weak compared to the "tail"
        iteration = 5
        bad_Bk = np.zeros((iteration + 1, iteration))
        np.fill_diagonal(bad_Bk, 1e-5) # Tiny singular values
        
        rule = self.RuleClass(self.tol, self.m, self.n)
        rule.update_regularizationparam(bad_Bk, self.b_norm)

        self.assertIsNotNone(rule.regalpha)
        self.assertTrue(rule.regalpha_low <= rule.regalpha <= rule.regalpha_high,
                        "Selected alpha under noise fell outside singular value bounds.")
        
class TestUpdateRegGCV(unittest.TestCase, RegRuleInfrastructureTestsMixin):
    """
    Unit tests for the GCV Regularization Parameter Selection Rule.
    """
    def setUp(self):
        '''Set up synthetic bidiagonal system for testing.'''
        self.setup_defaults(RuleClass=UpdateRegGCV)

    def test_gcv_minimization(self):
        '''Verify that the selected alpha minimizes the GCV functional.'''
        pass

    def test_wgcv_weighting_effect(self):
        '''Verify that the WGCV weighting affects the selected alpha as expected.'''
        pass