#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
import numpy as np
from cil.framework import ImageGeometry
from cil.optimisation.operators import (
    IdentityOperator,
    DiagonalOperator,
    GLSQROperator,
    LinearOperator,
    WaveletOperator,
)
from testclass import CCPiTestClass


class TestGLSQROperator(CCPiTestClass):
    def setUp(self):
        # Use a power-of-two size for Wavelet compatibility
        M = 16 
        self.ig = ImageGeometry(M, M)
        self.x = self.ig.allocate("random", seed=100)

        self.scaling_val = 2.0
        diag = self.ig.allocate(self.scaling_val)
        self.L_struct = DiagonalOperator(diag)

    def test_initialization(self):
        # Test default L2 initialization
        op = GLSQROperator(self.ig)
        self.assertEqual(op.norm_type, "L2")
        self.assertIsInstance(op.L_norm, IdentityOperator)
        self.assertIsInstance(op.L_struct, IdentityOperator)

        # Test L1 initialization
        tau = 0.5
        op_l1 = GLSQROperator(self.ig, norm_type="L1", tau=tau)
        self.assertEqual(op_l1.norm_type, "L1")
        self.assertIsInstance(op_l1.L_norm, DiagonalOperator)
        # Check initial weights: tau^-0.5
        expected_weight = tau**-0.5
        np.testing.assert_almost_equal(
            op_l1.L_norm.diagonal.as_array(), expected_weight
        )

        # Test validation of parameters
        with self.assertRaises(ValueError):
            GLSQROperator(self.ig, tau=-1)  # Negative tau
        with self.assertRaises(ValueError):
            GLSQROperator(self.ig, tau_factor=1.5)  # Factor out of range

    def test_input_validation(self):
        """Tests that the operator raises appropriate errors for invalid inputs."""

        # 1. Constructor Validation: tau
        with self.assertRaisesRegex(ValueError, "tau must be positive"):
            GLSQROperator(self.ig, tau=0)
        with self.assertRaisesRegex(ValueError, "tau must be positive"):
            GLSQROperator(self.ig, tau=-1.0)

        # 2. Constructor Validation: tau_mode
        with self.assertRaisesRegex(ValueError, "Unknown tau_mode"):
            GLSQROperator(self.ig, tau_mode="invalid_mode")

        # 3. Constructor Validation: tau_factor
        with self.assertRaisesRegex(ValueError, "tau_factor must be in the interval"):
            GLSQROperator(self.ig, tau_factor=1.0)
        with self.assertRaisesRegex(ValueError, "tau_factor must be in the interval"):
            GLSQROperator(self.ig, tau_factor=0.0)

        # 4. Constructor Validation: tau_min
        with self.assertRaisesRegex(
            ValueError, "tau_min must be positive and less than tau"
        ):
            GLSQROperator(self.ig, tau=1.0, tau_min=1.5)

        # 5. Constructor Validation: norm_type
        with self.assertRaisesRegex(ValueError, "Unknown norm_type"):
            GLSQROperator(self.ig, norm_type="L3")

        # 6. Operational Validation: Geometry mismatches
        op = GLSQROperator(self.ig, struct_operator=self.L_struct)
        wrong_ig = ImageGeometry(5, 5)
        x_wrong = wrong_ig.allocate()

        # Direct, Adjoint, and Inverse should fail if input geometry doesn't match
        with self.assertRaises(ValueError):
            op.direct(x_wrong)
        with self.assertRaises(ValueError):
            op.adjoint(x_wrong)
        with self.assertRaises(ValueError):
            op.inverse(x_wrong)

    def test_no_inverse_validation(self):
        """Tests validation that struct_operator must have an inverse method."""

        class MockOp(LinearOperator):
            def direct(self, x, out=None):
                return x

            def adjoint(self, x, out=None):
                return x

            # Missing inverse()

        uninvertible_op = MockOp(self.ig)
        with self.assertRaisesRegex(ValueError, "must have an 'inverse' method"):
            GLSQROperator(self.ig, struct_operator=uninvertible_op, norm_type="L2")

        with self.assertRaisesRegex(ValueError, "must have an 'inverse' method"):
            GLSQROperator(self.ig, struct_operator=uninvertible_op, norm_type="L1")

    def test_direct_adjoint_L2(self):
        # L2 case: L_tilde = I * L_struct = L_struct
        op = GLSQROperator(self.ig, struct_operator=self.L_struct, norm_type="L2")
        x = self.ig.allocate("random", seed=1)

        # Direct: L_norm(L_struct(x)) -> I(2.0 * x) = 2.0 * x
        res = op.direct(x)
        np.testing.assert_allclose(res.as_array(), self.scaling_val * x.as_array())

        # Adjoint: L_struct*(L_norm*(x)) -> 2.0 * I(x) = 2.0 * x
        res_adj = op.adjoint(x)
        np.testing.assert_allclose(res_adj.as_array(), self.scaling_val * x.as_array())

        # Dot test
        self.assertTrue(op.dot_test(op))

    def test_inverse(self):
        op = GLSQROperator(self.ig, struct_operator=self.L_struct, norm_type="L2")
        x = self.ig.allocate("random", seed=2)

        # Inverse: L_struct^-1 ( L_norm^-1 (x) )
        y = op.direct(x)
        x_rec = op.inverse(y)
        np.testing.assert_allclose(x_rec.as_array(), x.as_array(), atol=1e-5)

        op_l1 = GLSQROperator(
            self.ig, struct_operator=self.L_struct, norm_type="L1", tau=1.0
        )
        y_l1 = op_l1.direct(x)
        x_rec_l1 = op_l1.inverse(y_l1)
        np.testing.assert_allclose(x_rec_l1.as_array(), x.as_array(), atol=1e-5)

    def test_update_weights_L1(self):
        tau = 1.0
        # Set adapt_tau=False so tau stays exactly 1.0 for the math check
        op = GLSQROperator(self.ig, norm_type="L1", tau=tau, adapt_tau=False)

        # Create a "solution" x
        x = self.ig.allocate(3.0)

        # Calculate weights: w = (x^2 + tau^2)^-1/4
        op.update_weights(x)

        # Expected: (9 + 1)^-0.25 approx 0.562341
        expected_w = (3.0**2 + tau**2) ** -0.25
        np.testing.assert_allclose(op.L_norm.diagonal.as_array(), expected_w, rtol=1e-5)

    def test_out_parameters(self):
        # Check if 'out' parameter is correctly populated in direct and adjoint
        op = GLSQROperator(self.ig, struct_operator=self.L_struct)
        x = self.ig.allocate("random")
        out_dir = self.ig.allocate()
        out_adj = self.ig.allocate()
        out_inv = self.ig.allocate()

        op.direct(x, out=out_dir)
        op.adjoint(x, out=out_adj)
        op.inverse(x, out=out_inv)  # just to ensure no error

        np.testing.assert_allclose(out_dir.as_array(), op.direct(x).as_array())
        np.testing.assert_allclose(out_adj.as_array(), op.adjoint(x).as_array())
        np.testing.assert_allclose(out_inv.as_array(), op.inverse(x).as_array())

    def test_tau_properties(self):
        """Tests getting and setting of the tau property with validation."""
        op = GLSQROperator(self.ig, norm_type="L1", tau=1.0)

        # Test basic getter
        self.assertEqual(op.tau, 1.0)

        # Test basic setter
        op.tau = 0.5
        self.assertEqual(op.tau, 0.5)

        # Test validation in setter
        with self.assertRaisesRegex(ValueError, "tau must be positive"):
            op.tau = 0
        with self.assertRaisesRegex(ValueError, "tau must be positive"):
            op.tau = -0.1

    def test_tau_adaptation_logic(self):
        """Tests that adaptation logic correctly updates the tau property."""
        initial_tau = 1.0
        factor = 0.1
        op = GLSQROperator(
            self.ig,
            norm_type="L1",
            tau=initial_tau,
            adapt_tau=True,
            tau_mode="factor",
            tau_factor=factor,
        )

        # Manually trigger adaptation
        # The internal _adapt_tau should use the setter: self.tau = max(...)
        op._adapt_tau()

        self.assertAlmostEqual(op.tau, initial_tau * factor)

        # Test reaching tau_min
        op.tau_min = 0.05
        op._adapt_tau()  # 0.1 * 0.1 = 0.01, which is < tau_min
        self.assertEqual(op.tau, 0.05)

    def test_initialization_tau_sync(self):
        """Ensures constructor correctly initializes the property."""
        tau_val = 2.5
        op = GLSQROperator(self.ig, norm_type="L1", tau=tau_val)
        self.assertEqual(op._tau, tau_val)  # check internal
        self.assertEqual(op.tau, tau_val)  # check property access

    def test_wavelet_struct_operator(self):
        """Tests GLSQROperator with WaveletOperator as L_struct."""
        try:
            # Initialize WaveletOperator (Haar is orthogonal, so inverse=adjoint)
            W = WaveletOperator(self.ig, wavelet='haar', level=1)
        except ImportError:
            self.skipTest("PyWavelets not installed, skipping Wavelet test.")

        # 1. Test L2 Norm: L_tilde = I * W = W
        op = GLSQROperator(self.ig, struct_operator=W, norm_type="L2")
        
        # Test Direct/Inverse consistency
        # x_rec = W^-1( I^-1( I( W(x) ) ) )
        y = op.direct(self.x)
        x_rec = op.inverse(y)
        np.testing.assert_allclose(x_rec.as_array(), self.x.as_array(), atol=1e-5)

        # Dot test: Verify adjoint consistency
        # For orthogonal wavelets, W^* = W^-1. Since L_norm=I, 
        # the GLSQROperator itself should satisfy the dot test.
        self.assertTrue(op.dot_test(op))

        # 2. Test L1 Norm: L_tilde = D * W
        tau = 0.1
        op_l1 = GLSQROperator(self.ig, struct_operator=W, norm_type="L1", tau=tau)
        
        # Verify Inverse logic for L1
        # x_rec = W^-1( D^-1( y ) )
        y_l1 = op_l1.direct(self.x)
        x_rec_l1 = op_l1.inverse(y_l1)
        np.testing.assert_allclose(x_rec_l1.as_array(), self.x.as_array(), atol=1e-5)

    def test_inverse_with_scaling_struct(self):
        """Standard test for inverse with the setup's scaling operator."""
        op = GLSQROperator(self.ig, struct_operator=self.L_struct, norm_type="L2")
        x = self.ig.allocate("random", seed=2)

        # Inverse: L_struct^-1 ( L_norm^-1 (x) )
        y = op.direct(x)
        x_rec = op.inverse(y)
        np.testing.assert_allclose(x_rec.as_array(), x.as_array(), atol=1e-5)

    def test_inverse_adjoint_L2(self):
        """Test inverse_adjoint for L2 regularization: (L_struct^*)^-1"""
        # L2 case: L_tilde = I * L_struct
        op = GLSQROperator(self.ig, struct_operator=self.L_struct, norm_type="L2")
        
        # Test vector in the range of the operator
        y = op.range_geometry().allocate("random", seed=50)
        u = op.domain_geometry().allocate("random", seed=60)
        
        # Identity: <L_inv v, u> == <v, L_inv_adj u>
        # L_inv = L_struct^-1
        # L_inv_adj = (L_struct^*)^-1
        
        L_inv_v = op.inverse(y)
        L_inv_adj_u = op.inverse_adjoint(u)
        
        dot1 = L_inv_v.dot(u)
        dot2 = y.dot(L_inv_adj_u)
        
        self.assertAlmostEqual(dot1, dot2, places=5)

    def test_inverse_adjoint_L1(self):
        """Test inverse_adjoint for L1 weighting: (L_struct^* * L_norm^*)^-1"""
        tau = 0.5
        op = GLSQROperator(self.ig, struct_operator=self.L_struct, norm_type="L1", tau=tau)
        
        # y is in Range (weighted wavelet space), u is in Domain (Image space)
        y = op.range_geometry().allocate("random", seed=70)
        u = op.domain_geometry().allocate("random", seed=80)
        
        # Perform inverse and inverse_adjoint
        res_inv = op.inverse(y)
        res_inv_adj = op.inverse_adjoint(u)
        
        # Adjoint identity for the inverse
        dot1 = res_inv.dot(u)
        dot2 = y.dot(res_inv_adj)
        
        self.assertAlmostEqual(dot1, dot2, places=5)

    def test_wavelet_inverse_adjoint(self):
        """Verify GLSQROperator.inverse_adjoint with WaveletOperator."""
        try:
            from cil.optimisation.operators import WaveletOperator
            # Use Daubechies wavelet (not self-adjoint, but orthogonal)
            W = WaveletOperator(self.ig, wavelet='db2', level=1)
        except (ImportError, ModuleNotFoundError):
            self.skipTest("PyWavelets not installed, skipping Wavelet test.")

        op = GLSQROperator(self.ig, struct_operator=W, norm_type="L2")
        
        u = op.domain_geometry().allocate('random', seed=10)
        v = op.range_geometry().allocate('random', seed=20)
        
        # L_inv = W^-1
        # L_inv_adj = (W^*)^-1 = W (since W is orthogonal)
        L_inv_v = op.inverse(v)
        L_inv_adj_u = op.inverse_adjoint(u)
        
        dot1 = L_inv_v.dot(u)
        dot2 = v.dot(L_inv_adj_u)
        
        # Normalize to ensure precision check is scale-invariant
        scale = u.norm() * v.norm()
        self.assertAlmostEqual(dot1/scale, dot2/scale, places=5)

    def test_inverse_adjoint_out(self):
        """Verify the 'out' parameter in inverse_adjoint."""
        op = GLSQROperator(self.ig, struct_operator=self.L_struct)
        u = self.ig.allocate("random")
        out_inv_adj = self.ig.allocate()
        
        op.inverse_adjoint(u, out=out_inv_adj)
        expected = op.inverse_adjoint(u)
        
        np.testing.assert_allclose(out_inv_adj.as_array(), expected.as_array())


if __name__ == "__main__":
    unittest.main()
