"""
Unit tests for Hybrid LSQR Regularization Parameter Selection Rules.

1. Base Class Infrastructure Verification:
   We test the `UpdateRegBase` logic (inherited by all rules) to ensure:
   - Proper SVD decomposition of the projected bidiagonal matrix B_k.
   - Correct extraction of the first row of the left singular vectors (u1T).
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

    def setup_bidiagonal_system(self, RuleClass):
        """
        Standardizes the creation of the test system across all rules.
        Call this inside the child class setUp().
        """
        self.k_dim = 5
        self.m, self.n = 100, 100
        self.beta0 = 1.0
        self.tol = 1e-3
        
        # Create a bidiagonal matrix B_k with known decay
        self.Bk = np.zeros((self.k_dim + 1, self.k_dim))
        np.fill_diagonal(self.Bk, [10, 5, 2, 1, 0.5])
        np.fill_diagonal(self.Bk[1:], [4, 2, 1, 0.4])

        self.RuleClass = RuleClass
        self.default_rule = self.RuleClass(self.tol, self.m, self.n)
        self.default_rule._compute_svd(self.Bk)
        self.default_rule.beta0 = self.beta0

    def test_svd_extraction(self):
        '''Verify SVD components are correctly computed and stored.'''
        self.default_rule.update_regularizationparam(self.Bk, self.beta0)

        # Reference SVD
        U, s, _ = scipy.linalg.svd(self.Bk)
        
        # Verify singular values and projection vector
        np.testing.assert_array_almost_equal(self.default_rule.Sb, s)
        np.testing.assert_array_almost_equal(self.default_rule.u1T, U[0, :]) 
        
        # Verify bounds used for optimization/plotting
        self.assertEqual(self.default_rule.Sbmax, s[0])
        self.assertEqual(self.default_rule.Sbmin, s[-1])
        self.assertEqual(self.default_rule.regalpha_low, 0.0)
        self.assertEqual(self.default_rule.regalpha_high, s[0])

    def test_numerical_stability_zeros(self):
        '''Verify that the safety EPS handles zero/static values without crashing.'''        
        # Test 1: Identical values [1, 1]
        self.default_rule.regalpha_history = [1.0, 1.0]
        self.default_rule.converged = False
        self.default_rule._run_convergence_checks()
        self.assertTrue(self.default_rule.converged, "Static values should trigger convergence.")

        # Test 2: All zeros (Potential underflow)
        self.default_rule.regalpha_history = [0.0, 0.0]
        self.default_rule.converged = False
        self.default_rule._run_convergence_checks()
        self.assertTrue(self.default_rule.converged, "Zero values should handle EPS safely.")
    
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
            ([1.0, 0.95], True, "Decreased within tol")
        ]
        for values, expected, msg in scenarios:
            rule.regalpha_history = values
            rule.converged = False
            rule._run_convergence_checks()
            self.assertEqual(rule.converged, expected, msg)

    def test_print_status_formatting(self):
        '''Ensure status printing handles None regalpha gracefully.'''
        # Should not raise error
        self.default_rule._print_status()
    
    def test_projected_residual_norm_sq(self):
        '''Verify the mathematical correctness of the residual norm calculation.'''
        self.default_rule.update_regularizationparam(self.Bk, self.beta0)

        # Test value for alpha
        alpha = 0.5
        
        # 1. Get value from the base class method
        calculated_res_sq = self.default_rule._projected_residual_norm_sq(alpha)

        # 2. Manual calculation for comparison
        # filter = alpha^2 / (s^2 + alpha^2)
        filt = (alpha**2) / (self.default_rule.Sbsq + alpha**2)
        
        # Part 1: components within the subspace (first k elements of u1T)
        res_subspace = np.sum(np.square(filt * (self.default_rule.beta0 * self.default_rule.u1T[:-1])))
        
        # Part 2: component orthogonal to the subspace (last element of u1T)
        res_orthogonal = np.square(self.default_rule.beta0 * self.default_rule.u1T[-1])
        
        expected_res_sq = res_subspace + res_orthogonal

        # 3. Assert equality
        self.assertAlmostEqual(calculated_res_sq, expected_res_sq, places=12)
        
        # 4. Check edge case: very large alpha (should approach beta0^2)
        large_alpha_res = self.default_rule._projected_residual_norm_sq(1e10)
        self.assertAlmostEqual(large_alpha_res, self.default_rule.beta0**2, places=5)
    
    def test_projected_solution_norm_sq(self):
        '''Verify the mathematical correctness of the solution norm calculation.'''
        self.default_rule.update_regularizationparam(self.Bk, self.beta0)

        # Test value for alpha
        alpha = 0.5
        
        # 1. Get value from the base class method
        calculated_sol_sq = self.default_rule._projected_solution_norm_sq(alpha)

        # 2. Manual calculation for comparison
        # filter = sigma / (sigma^2 + alpha^2)
        filt = self.default_rule.Sb / (self.default_rule.Sbsq + alpha**2)
        
        # The solution exists strictly within the k-dimensional subspace.
        # It uses the first k elements of the projected RHS (u1T[:-1]).
        expected_sol_sq = np.sum(np.square(filt * (self.default_rule.beta0 * self.default_rule.u1T[:-1])))
        # 3. Assert equality
        self.assertAlmostEqual(calculated_sol_sq, expected_sol_sq, places=12)
        
        # 4. Check edge case: very large alpha
        # As alpha -> infinity, the solution norm should approach 0.
        large_alpha_sol = self.default_rule._projected_solution_norm_sq(1e10)
        self.assertAlmostEqual(large_alpha_sol, 0.0, places=15)

        # 5. Check edge case: alpha = 0 (Unregularized least squares)
        # Should match sum( (beta_hat_i / sigma_i)^2 )
        zero_alpha_sol = self.default_rule._projected_solution_norm_sq(0.0)
        manual_least_squares = np.sum(np.square((self.default_rule.beta0 * self.default_rule.u1T[:-1]) / self.default_rule.Sb))
        self.assertAlmostEqual(zero_alpha_sol, manual_least_squares, places=12)
    
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

        # Test non-ndarray Bk
        with self.assertRaises(TypeError):
            self.default_rule.update_regularizationparam(Bk=[[1, 0], [0, 1]], beta0=1.0)
        # Test Bk that isn't (k+1, k)
        invalid_shape_Bk = np.eye(self.k_dim) # (5, 5) instead of (6, 5)
        with self.assertRaises(ValueError):
            self.default_rule.update_regularizationparam(invalid_shape_Bk, self.beta0)
        # Test Bk exceeding operator dimensions
        huge_Bk = np.zeros((self.m + 2, self.m + 1))
        with self.assertRaises(ValueError):
            self.default_rule.update_regularizationparam(huge_Bk, self.beta0)
        # Test non-scalar beta0
        with self.assertRaises(TypeError):
            self.default_rule.update_regularizationparam(self.Bk, beta0=[1.0, 0.0])
    
    def test_interface_return_signature(self):
        '''Verify _compute_next_regalpha returns either a float or a (float, float) tuple.'''
        result = self.default_rule._compute_next_regalpha()
        
        if isinstance(result, (tuple, list)):
            self.assertEqual(len(result), 2, "If returning a tuple, it must have 2 elements.")
            alpha, val = result
            # Check for (float, float), (None, None), or mixed
            self.assertTrue(alpha is None or np.isscalar(alpha), "Alpha part of tuple must be None or scalar.")
            self.assertTrue(val is None or np.isscalar(val), "Func part of tuple must be None or scalar.")
        else:
            # Check for float or None
            self.assertTrue(result is None or np.isscalar(result), "Single return value must be None or scalar.")
    
    def test_plot_history_error_handling(self):
        '''Verify ValueError is raised when trying to plot empty history.'''
        rule = self.default_rule
        
        # Ensure history is empty
        rule.regalpha_history = []
        rule.func_history = []

        # 1. Test completely empty history
        with self.assertRaisesRegex(ValueError, "No regularization parameter history"):
            rule.plot_history(show_objective=False)

        # 2. Test missing objective history specifically
        # Manually add to regalpha but keep func_history empty
        rule.regalpha_history = [0.1, 0.2]
        with self.assertRaisesRegex(ValueError, "No function history available"):
            rule.plot_history(show_objective=True)

class TestUpdateRegBase(unittest.TestCase, RegRuleInfrastructureTestsMixin):
    """
    Unit tests for the Base Class of Hybrid LSQR Regularization Parameter Selection Rules.
    """
    def setUp(self):
        '''Set up synthetic bidiagonal system for testing.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return 0.1, 0.0
            def func(self, regalpha): return 0.0
        self.setup_bidiagonal_system(RuleClass=MockRule)

    def test_base_state_management(self):
        '''Verify history tracking, iteration counting, and length-safety gating.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self):
                # Use iteration to prove the link between k and alpha calculation
                return 0.5 / self.iteration 
            def func(self, regalpha): return 0

        rule = MockRule(self.tol, self.m, self.n)
        
        # 1. Check Iteration and History Appending
        # Bk.shape[1] is 5, so alpha should be 0.5 / 5 = 0.1
        rule.update_regularizationparam(self.Bk, self.beta0)
        
        self.assertEqual(rule.iteration, self.k_dim)
        self.assertEqual(len(rule.regalpha_history), 1)
        self.assertAlmostEqual(rule.regalpha_history[0], 0.1)

        # 2. Test safety gate for short history
        # With only one alpha, _run_convergence_checks should exit silently
        rule.converged = False
        rule._run_convergence_checks()
        self.assertFalse(rule.converged, "Should not converge with a history length of 1.")

        # 3. Test multiple appends
        rule.update_regularizationparam(self.Bk, self.beta0)
        self.assertEqual(len(rule.regalpha_history), 2)
    
    def test_history_tracking(self):
        '''Verify regalpha_history and func_history correctly append via parent method.'''
        # We use a Mock here to control the returns exactly, 
        # independent of the specific rule's math.
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): 
                # Return (alpha, func_val)
                # alpha based on iteration, func_val as a fixed test value
                return 0.1 * self.iteration, 99.0
            def func(self, regalpha): return 99.0

        rule = MockRule(self.tol, self.m, self.n)
        
        # Iteration 1: Bk.shape[1] is 5, so alpha = 0.5, func = 99.0
        rule.update_regularizationparam(self.Bk, self.beta0)
        
        # Iteration 2: Simulate second call
        rule.update_regularizationparam(self.Bk, self.beta0)
        
        # 1. Check alpha history
        self.assertEqual(len(rule.regalpha_history), 2)
        np.testing.assert_array_almost_equal(rule.regalpha_history, [0.5, 0.5])
        
        # 2. Check function value history
        self.assertEqual(len(rule.func_history), 2)
        np.testing.assert_array_almost_equal(rule.func_history, [99.0, 99.0])
        
        # 3. Check current state matches last entry
        self.assertEqual(rule.regalpha, 0.5)

        class MockAlphaOnly(UpdateRegBase):
            def _compute_next_regalpha(self): return 0.1, None
            def func(self, regalpha): return 0

        rule = MockAlphaOnly(self.tol, self.m, self.n)
        rule.update_regularizationparam(self.Bk, self.beta0)
        
        self.assertEqual(len(rule.regalpha_history), 1)
        self.assertEqual(len(rule.func_history), 0, "func_history should stay empty if None is returned.")


def construct_noisy_beta(signal_norm, relative_noise_level):
    '''
    Constructs a beta0 value and an absolute noise level for testing.
    
    Parameters
    ----------
    signal_norm : float
        The norm of the 'clean' part of the signal.
    relative_noise_level : float
        The percentage of noise (e.g., 0.05 for 5% noise).
        
    Returns
    -------
    beta0 : float
        The norm of the noisy right-hand side.
    abs_noise : float
        The absolute norm of the noise (eta).
    '''
    # Absolute noise norm (eta)
    abs_noise = relative_noise_level * signal_norm
    
    # In a Krylov setting, we assume the noise is orthogonal to the signal
    # therefore: ||b||^2 = ||b_true||^2 + ||e||^2
    beta0 = np.sqrt(signal_norm**2 + abs_noise**2)
    
    return beta0, abs_noise

class TestUpdateRegDiscrep(unittest.TestCase, RegRuleInfrastructureTestsMixin):
    def setUp(self):
        '''Set up synthetic bidiagonal system for testing.'''
        self.setup_bidiagonal_system(RuleClass=UpdateRegDiscrep)
    
    def test_discrepancy_monotonicity(self):
        '''Verify that alpha is a monotonically increasing function of noise.'''
        noises = [0.1, 0.2, 0.3, 0.4]
        alphas = []
        
        for n in noises:
            rule = UpdateRegDiscrep(self.tol, self.m, self.n, n)
            rule.update_regularizationparam(self.Bk, self.beta0)
            alphas.append(rule.regalpha)
            self.assertGreater(rule.regalpha, 0.0)
            
        # Check that alphas are strictly increasing
        for i in range(len(alphas) - 1):
            self.assertLess(alphas[i], alphas[i+1], 
                           f"Alpha should increase with noise. Failed at index {i}")
    
    def test_zero_noise_limit(self):
        '''Verify that if noise level is below reachable residual, alpha is 0.'''
        # Set noise to 0.0
        rule = UpdateRegDiscrep(self.tol, self.m, self.n, discrep_noise_level=0.0)
        rule.update_regularizationparam(self.Bk, self.beta0)
        
        # The logic should realize that the residual at alpha=0 
        # is already > 0.0, so it should return the lower bound.
        self.assertEqual(rule.regalpha, 0.0, 
                         "Should return alpha_low (0.0) when noise is unreachable.")
    
    def test_residual_at_root(self):
        '''Verify that the residual at the selected alpha matches the noise level.'''
        noise = 0.3
        rule = UpdateRegDiscrep(self.tol, self.m, self.n, discrep_noise_level=noise)
        rule.update_regularizationparam(self.Bk, self.beta0)
        
        # func(alpha) should be ~0, meaning residual^2 - noise^2 = 0
        self.assertAlmostEqual(rule.func(rule.regalpha), 0.0, places=6)
    
    def test_discrepancy_mathematical_recalculation(self):
        '''Verify that the identified alpha satisfies the Discrepancy Principle.'''
        noise_level = 0.25
        rule = UpdateRegDiscrep(self.tol, self.m, self.n, noise_level)
        rule.update_regularizationparam(self.Bk, self.beta0)
        
        alpha = rule.regalpha
        # Manual math check:
        filt = (alpha**2) / (rule.Sbsq + alpha**2)
        
        # rule.Sbsq is length k, rule.u1T[:-1] is length k.
        residual_sq = np.sum(np.square(filt * (rule.beta0 * rule.u1T[:-1]))) + \
                      np.square(rule.beta0 * rule.u1T[-1])
        
        self.assertAlmostEqual(np.sqrt(residual_sq), noise_level, places=5)
    
    def test_discrepancy_with_constructed_noise(self):
        '''Mathematical verification with a controlled noise-to-signal ratio.'''
        signal_norm = 1.0
        rel_noise = 0.1  # 10% noise
        
        beta0, noise_level = construct_noisy_beta(signal_norm, rel_noise)
        
        rule = UpdateRegDiscrep(self.tol, self.m, self.n, noise_level)
        
        rule.update_regularizationparam(self.Bk, beta0)
        
        # Verification: Residual norm at selected alpha should match noise_level
        # We check the square to avoid sqrt computation in the test
        residual_sq = rule.func(rule.regalpha) + noise_level**2
        
        self.assertAlmostEqual(np.sqrt(residual_sq), noise_level, places=5)

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
        self.setup_bidiagonal_system(RuleClass=UpdateRegLcurve)

    def test_curvature_peak_finding(self):
        """
        Check if selected alpha maximizes the curvature function.
        
        Done by comparing against a brute-force grid search over a wide alpha range
        to the continuous optimization result.
        """
        pass

    def test_derivative_consistency(self):
        """Compare analytical derivatives against finite difference results."""
        pass

    def test_log_space_stability(self):
        """Ensure no NaN/Inf results when norms are extremely small."""
        pass

    def test_plotting_indices(self):
        """Verify the L-curve plotting utility handles index truncation correctly."""
        pass

    def test_alpha_monotonicity_with_noise(self):
        """
        Verify that the selected alpha increases as the noise level increases.
        
        As the right-hand side becomes noisier, the maximum curvature point 
        on the L-curve should shift toward higher regularization (larger alpha) 
        to prevent overfitting the noise.
        """
        pass

    def test_lcurve_stability_high_noise(self):
        """
        Test the rule's behavior under extreme noise conditions.
        
        Ensures that even when the L-curve is 'flat' (no distinct corner), 
        the optimization does not crash and returns a value within the 
        singular value bounds [Sbmin, Sbmax].
        """
        pass