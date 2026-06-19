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

from cil.framework import BlockGeometry
from cil.optimisation.operators import LinearOperator, IdentityOperator, BlockOperator
from Wrappers.Python.cil.optimisation.operators.DiagonalOperator import DiagonalOperator
from Wrappers.Python.cil.optimisation.operators.GradientOperator import GradientOperator

import logging
import numpy as np

log = logging.getLogger(__name__)



class HybridTikhonovOperator(LinearOperator):
    r"""
    Standard Form Transformation Operator for Hybrid LSQR.

    Transforms the generalised Tikhonov problem into standard form by mapping 
    the search space from the physical domain to the regularised structure space.

    $$ K = A L_{\text{weighted}}^{-1} = A (W L)^{-1} $$

    Because the LSQR algorithm natively handles the Tikhonov penalty ($\alpha I$) 
    via internal scalar damping, this operator does not require a block formulation. 
    It purely handles the spatial transformations.

    By transforming the operator, the solver operates in an intermediate 
    regularised space ($x$). Once the inner Krylov solver finds the optimal $x$, 
    the true physical solution must be recovered via the inverse mapping 
    $u = L_{\text{weighted}}^{-1} x$. 
    
    The exact mechanics of this recovery mapping are delegated entirely to the 
    underlying regularisation operator (e.g., `WeightedStructOperator`).

    Geometries
    ----------
    Domain: Range($L$) -> The regularised structure space ($x$).
    Range:  Range($A$) -> The physical data space ($b$).
    """

    def __init__(
        self,
        operator,
        solution_geometry,
        struct_operator=None,
        tmp_domain=None, tmp_range_struct=None,
    ):
        if struct_operator is None:
            struct_operator = IdentityOperator(solution_geometry)
            
        self.operator = operator
        self.struct_operator = struct_operator
        
        # Set up Weighted Structural Operator
        self.reg_operator = WeightedStructOperator(
            domain_geometry=solution_geometry,
            struct_operator=struct_operator,
            tmp_range_struct=tmp_range_struct,
        )

        hybrid_domain = struct_operator.range_geometry()
        hybrid_range = operator.range_geometry()
        
        self.tmp_domain = tmp_domain if tmp_domain is not None else solution_geometry.allocate()

        super(HybridTikhonovOperator, self).__init__(
            domain_geometry=hybrid_domain, 
            range_geometry=hybrid_range
        )

    @property
    def weights(self):
        """Get the current diagonal weights used for IRLS."""
        if isinstance(self.reg_operator.weight_operator, DiagonalOperator):
            return self.reg_operator.weight_operator.diagonal
        raise AttributeError("Weight operator is not a DiagonalOperator; cannot retrieve weights.")
    
    @weights.setter
    def weights(self, new_weights):
        """Set new diagonal weights to approximate the L1 norm."""
        self.reg_operator.update_weights(new_weights)

    def direct(self, x, out=None):
        r"""
        Applies $K x = A (WL)^{-1} x$
        """
        if out is None:
            out = self.range_geometry().allocate()

        self.reg_operator.inverse(x, out=self.tmp_domain)
        self.operator.direct(self.tmp_domain, out=out)
        return out

    def adjoint(self, y, out=None):
        r"""
        Applies $K^* y = (WL)^{-*} A^* y$
        """
        if out is None:
            out = self.domain_geometry().allocate()

        self.operator.adjoint(y, out=self.tmp_domain)
        self.reg_operator.inverse_adjoint(self.tmp_domain, out=out)
        return out


class BlockTikhonovOperator(BlockOperator):
    r"""
    Block Tikhonov Operator for general L2 and L1 regularisation.

    This block operator performs analysis regularisation directly in the 
    physical solution space ($u$). By natively inheriting from CIL's 
    `BlockOperator`, it seamlessly handles the block algebra for:

    $$ K = \begin{bmatrix} A \\ \alpha W L \end{bmatrix} $$

    It should be the default choice for `CGLS` or standard `LSQR` when hybrid 
    parameter selection is NOT being used.

    Geometries
    ----------
    Domain: Domain($A$) -> The physical solution space ($u$).
    Range:  BlockGeometry(Range($A$), Range($L$))
    """
    def __init__(
        self,
        operator,
        solution_geometry,
        struct_operator=None,
        regalpha: float = 0.0,
        tmp_range_struct=None,
    ):
        if struct_operator is None:
            struct_operator = IdentityOperator(solution_geometry)
            
        self.operator = operator
        self.struct_operator = struct_operator
        
        self.reg_operator = WeightedStructOperator(
            domain_geometry=solution_geometry,
            struct_operator=struct_operator,
            tmp_range_struct=tmp_range_struct,
        )

        self.scaled_reg_operator = regalpha * self.reg_operator

        # Initialize CIL's native BlockOperator with A and \alpha WL
        super(BlockTikhonovOperator, self).__init__(
            self.operator, self.scaled_reg_operator
        )

    @property
    def regalpha(self):
        """Get the current Tikhonov regularisation parameter."""
        return self.scaled_reg_operator.scalar

    @regalpha.setter
    def regalpha(self, value: float):
        """Dynamically update the regularisation parameter."""
        self.scaled_reg_operator.scalar = value

    @property
    def weights(self):
        """Get the current diagonal weights used for IRLS."""
        if isinstance(self.reg_operator.weight_operator, DiagonalOperator):
            return self.reg_operator.weight_operator.diagonal
        raise AttributeError("Weight operator is not a DiagonalOperator; cannot retrieve weights.")
    
    @weights.setter
    def weights(self, new_weights):
        """Set new diagonal weights to approximate the L1 norm."""
        self.reg_operator.update_weights(new_weights)
        

class WeightedStructOperator(LinearOperator):
    r"""
    Weighted Structural Operator for L1 Iteratively Reweighted Least Squares (IRLS).

    Composes a structural regularisation operator $L$ with a diagonal 
    weight matrix $W$ to form the effective regularisation operator
    
    $$ L_{\text{weighted}} = W L $$

    During IRLS, the weight matrix $W$ is dynamically updated to approximate 
    the L1 norm: $W_k = (u_{k-1}^{1/2} + \tau_k^{1/2})^{-1/4}$.

    **Analysis vs. Synthesis Recovery Mapping**
    When used inside a synthesis framework (like `HybridTikhonovOperator`), the 
    solver operates in the structure space ($x$) and relies on this class to map 
    the result back to the physical solution space ($u$) via its `inverse` method.

    For orthogonal structural operators (e.g., Wavelets), the analysis and synthesis 
    formulations are mathematically equivalent. The inverse is trivial, and the 
    exact physical solution is easily recovered.

    However, for rectangular or non-orthogonal operators (like Gradients / Finite 
    Differences), analysis and synthesis formulations are **not equivalent**. To 
    compute the exact physical solution mapped from the structure space, one would 
    need to compute the Moore-Penrose pseudo-inverse:

    $$ (WL)^{\dagger} = (L^T W^2 L)^{-1} L^T W $$

    Computing this exact pseudo-inverse is computationally expensive as it requires 
    an embedded linear solver (e.g., CGLS, multigrid) just for the recovery step. 
    Therefore, this class approximates the recovery by strictly chaining the available 
    inverses (or pseudo-inverses) provided by the operator:

    $$ u \approx L^{-1} W^{-1} x $$

    Consequently, this operator requires $L$ to have a strictly defined `inverse` 
    and `inverse_adjoint` method to handle this mapping.

    Geometries
    ----------
    Domain: Domain($u$) -> Solution space.
    Range:  Range($L$)  -> Weighted structure space.
    """
    def __init__(
        self,
        domain_geometry,
        struct_operator=None,
        tmp_range_struct=None,
    ):
        if struct_operator is not None:
            self.struct_operator = struct_operator
        else:
            self.struct_operator = IdentityOperator(domain_geometry)
            
        range_geometry = self.struct_operator.range_geometry()
        
        if isinstance(self.struct_operator, GradientOperator):
            if self.struct_operator.operator.bnd_cond != 'Dirichlet':
                raise ValueError(
                    "HybridTikhonovOperator requires GradientOperator with Dirichlet "
                    "boundary conditions due to null-space properties."
                )
        
        # Initialize as unweighted (L2)
        self.weight_operator = IdentityOperator(domain_geometry=range_geometry)
        self.tmp_range_struct = tmp_range_struct if tmp_range_struct is not None else range_geometry.allocate()

        super(WeightedStructOperator, self).__init__(
            domain_geometry=domain_geometry, range_geometry=range_geometry
        )

    def direct(self, x, out=None):
        r"""
        Returns $L_{\text{weighted}}(x) = W(L(x))$
        """
        if out is None:
            temp = self.struct_operator.direct(x)
            return self.weight_operator.direct(temp)
        else:
            self.struct_operator.direct(x, out=out)
            return self.weight_operator.direct(out, out=out)

    def adjoint(self, x, out=None):
        r"""
        Returns the adjoint $L_{\text{weighted}}^*(x) = L^*(W^*(x))$
        """
        self.weight_operator.adjoint(x, out=self.tmp_range_struct)
        if out is None:
            return self.struct_operator.adjoint(self.tmp_range_struct)
        
        self.struct_operator.adjoint(self.tmp_range_struct, out=out)
        return out

    def inverse(self, x, out=None):
        r"""
        Returns the inverse $L_{\text{weighted}}^{-1}(x) = L^{-1}(W^{-1}(x))$
        """
        self.weight_operator.inverse(x, out=self.tmp_range_struct)
        
        if out is None:
            return self.struct_operator.inverse(self.tmp_range_struct)
        else:
            self.struct_operator.inverse(self.tmp_range_struct, out=out)
            return out
        
    def inverse_adjoint(self, x, out=None):
        r"""
        Returns the adjoint of the inverse $L_{\text{weighted}}^{-*}(x) = W^{-*}(L^{-*}(x))$
        """
        if out is None:
            out = self.domain_geometry().allocate()

        self.struct_operator.inverse_adjoint(x, out=out)
        self.weight_operator.inverse_adjoint(out, out=out)
        return out

    def update_weights(self, weights):
        """
        Replace the current weights with a new diagonal weight operator.
        """
        if not isinstance(self.weight_operator, DiagonalOperator):
            self.weight_operator = DiagonalOperator(
                weights, 
                domain_geometry=self.range_geometry()
            )
        else:
            self.weight_operator.diagonal = weights