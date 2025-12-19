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
from cil.optimisation.utilities.HybridUpdateReg import UpdateRegBase, UpdateRegDiscrep
# (
#     UpdateRegBase, UpdateRegLcurve, UpdateRegGCV, 
#     UpdateRegWGCV, UpdateRegDiscrepancy
# )


class TestUpdateRegBase(unittest.TestCase):
    def setUp(self):
        '''Set up synthetic bidiagonal system for testing.'''
        self.k_dim = 5
        # Dimensions of the full problem
        self.m, self.n = 100, 100
        # Create a bidiagonal matrix B_k with known decay
        self.Bk = np.zeros((self.k_dim + 1, self.k_dim))
        np.fill_diagonal(self.Bk, [10, 5, 2, 1, 0.5])
        np.fill_diagonal(self.Bk[1:], [4, 2, 1, 0.4])
        
        self.beta0 = 1.0
        self.tol = 1e-3
    
    def test_svd_extraction(self):
        '''Verify SVD components are correctly computed and stored.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return 0.1
            def func(self, regalpha): return 0

        rule = MockRule(self.tol, self.m, self.n)
        rule.update_regularizationparam(self.Bk, self.beta0)

        # Reference SVD
        U, s, _ = scipy.linalg.svd(self.Bk)
        
        # Verify singular values and projection vector
        np.testing.assert_array_almost_equal(rule.Sb, s)
        np.testing.assert_array_almost_equal(rule.u1T, U[0, :]) 
        
        # Verify bounds used for optimization/plotting
        self.assertEqual(rule.Sbmax, s[0])
        self.assertEqual(rule.Sbmin, s[-1])
        self.assertEqual(rule.regalpha_low, 0.0)
        self.assertEqual(rule.regalpha_high, s[0])

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
        self.assertEqual(len(rule.regalphavec), 1)
        self.assertAlmostEqual(rule.regalphavec[0], 0.1)

        # 2. Test safety gate for short history
        # With only one alpha, _run_convergence_checks should exit silently
        rule.converged = False
        rule._run_convergence_checks()
        self.assertFalse(rule.converged, "Should not converge with a history length of 1.")

        # 3. Test multiple appends
        rule.update_regularizationparam(self.Bk, self.beta0)
        self.assertEqual(len(rule.regalphavec), 2)

    def test_numerical_stability_zeros(self):
        '''Verify that the safety EPS handles zero/static values without crashing.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return 1.0
            def func(self, regalpha): return 0
        rule = MockRule(self.tol, self.m, self.n)
        
        # Test 1: Identical values [1, 1]
        rule.regalphavec = np.array([1.0, 1.0])
        rule.converged = False
        rule._run_convergence_checks()
        self.assertTrue(rule.converged, "Static values should trigger convergence.")

        # Test 2: All zeros (Potential underflow)
        rule.regalphavec = np.array([0.0, 0.0])
        rule.converged = False
        rule._run_convergence_checks()
        self.assertTrue(rule.converged, "Zero values should handle EPS safely.")
    
    def test_convergence_logic(self):
        '''Verify convergence logic based on relative change in regalpha.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return 1.0
            def func(self, regalpha): return 0
        tol = 0.1
        rule = MockRule(tol, self.m, self.n)

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
            rule.regalphavec = np.array(values)
            rule.converged = False
            rule._run_convergence_checks()
            self.assertEqual(rule.converged, expected, msg)

    def test_print_status_formatting(self):
        '''Ensure status printing handles None regalpha gracefully.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return None
            def func(self, regalpha): return 0
            
        rule = MockRule(self.tol, self.m, self.n)
        # Should not raise error
        rule._print_status()
    
    def test_history_tracking(self):
        '''Verify regalphavec correctly appends new values via the parent method.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return 0.1 * self.iteration
            def func(self, regalpha): return 0

        rule = MockRule(self.tol, self.m, self.n)
        
        # Call update twice
        rule.update_regularizationparam(self.Bk, self.beta0) # k=5
        rule.update_regularizationparam(self.Bk, self.beta0) # k=5 (simulated)
        
        self.assertEqual(len(rule.regalphavec), 2)
        np.testing.assert_array_almost_equal(rule.regalphavec, [0.5, 0.5])
    
    def test_projected_residual_norm_sq(self):
        '''Verify the mathematical correctness of the residual norm calculation.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return 0.1
            def func(self, regalpha): return 0

        rule = MockRule(self.tol, self.m, self.n)
        rule.update_regularizationparam(self.Bk, self.beta0)

        # Test value for alpha
        alpha = 0.5
        
        # 1. Get value from the base class method
        calculated_res_sq = rule._projected_residual_norm_sq(alpha)

        # 2. Manual calculation for comparison
        # filter = alpha^2 / (s^2 + alpha^2)
        filt = (alpha**2) / (rule.Sbsq + alpha**2)
        
        # Part 1: components within the subspace (first k elements of u1T)
        res_subspace = np.sum(np.square(filt * (rule.beta0 * rule.u1T[:-1])))
        
        # Part 2: component orthogonal to the subspace (last element of u1T)
        res_orthogonal = np.square(rule.beta0 * rule.u1T[-1])
        
        expected_res_sq = res_subspace + res_orthogonal

        # 3. Assert equality
        self.assertAlmostEqual(calculated_res_sq, expected_res_sq, places=12)
        
        # 4. Check edge case: very large alpha (should approach beta0^2)
        large_alpha_res = rule._projected_residual_norm_sq(1e10)
        self.assertAlmostEqual(large_alpha_res, self.beta0**2, places=5)

    def test_abstract_enforcement(self):
        '''Ensure UpdateRegBase cannot be instantiated without abstract methods.'''
        with self.assertRaises(TypeError):
            UpdateRegBase('base', 1e-3, 10, 10)
    
    def test_invalid_initialization(self):
        '''Verify that __init__ rejects non-positive dimensions and tolerances.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return 0.1
            def func(self, regalpha): return 0

        # Test negative/zero dimensions
        with self.assertRaises(ValueError):
            MockRule(regalpha_saturation_tol=1e-3, m=0, n=100)
        with self.assertRaises(ValueError):
            MockRule(regalpha_saturation_tol=1e-3, m=100, n=-5)
        
        # Test non-positive tolerance
        with self.assertRaises(ValueError):
            MockRule(regalpha_saturation_tol=0, m=100, n=100)
        with self.assertRaises(ValueError):
            MockRule(regalpha_saturation_tol=-1e-4, m=100, n=100)

    def test_invalid_update_inputs(self):
        '''Verify that update_regularizationparam rejects inconsistent matrix shapes and types.'''
        class MockRule(UpdateRegBase):
            def _compute_next_regalpha(self): return 0.1
            def func(self, regalpha): return 0

        rule = MockRule(self.tol, self.m, self.n)

        # Test non-ndarray Bk
        with self.assertRaises(TypeError):
            rule.update_regularizationparam(Bk=[[1, 0], [0, 1]], beta0=1.0)

        # Test Bk that isn't (k+1, k)
        invalid_shape_Bk = np.eye(self.k_dim) # (5, 5) instead of (6, 5)
        with self.assertRaises(ValueError):
            rule.update_regularizationparam(invalid_shape_Bk, self.beta0)

        # Test Bk exceeding operator dimensions
        huge_Bk = np.zeros((self.m + 2, self.m + 1))
        with self.assertRaises(ValueError):
            rule.update_regularizationparam(huge_Bk, self.beta0)

        # Test non-scalar beta0
        with self.assertRaises(TypeError):
            rule.update_regularizationparam(self.Bk, beta0=[1.0, 0.0])


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

class TestUpdateRegDiscrep(unittest.TestCase):
    def setUp(self):
        '''Set up synthetic bidiagonal system for testing.'''
        self.k_dim = 5
        # Dimensions of the full problem
        self.m, self.n = 100, 100
        # Create a bidiagonal matrix B_k with known decay
        self.Bk = np.zeros((self.k_dim + 1, self.k_dim))
        np.fill_diagonal(self.Bk, [10, 5, 2, 1, 0.5])
        np.fill_diagonal(self.Bk[1:], [4, 2, 1, 0.4])
        
        self.beta0 = 1.0
        self.tol = 1e-3
    
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