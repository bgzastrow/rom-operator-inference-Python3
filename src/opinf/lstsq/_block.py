# lstsq/_bloc.py
"""Operator Inference least-squares solvers for block-structured systems."""

__all__ = [
    "BlockPlainSolver"
    "BlockTikhonovSolver"
]

import abc
# import warnings
import numpy as np
import scipy.linalg as la
# import scipy.sparse as sparse

from ._base import _BaseSolver


# Solver classes ==============================================================
class BlockPlainSolver(_BaseSolver):
    """Solve the l2-norm ordinary least-squares problem without any
    regularization, i.e.,

        min_{X} ||AX - B||_F^2.

    The solution is calculated using scipy.linalg.lstsq().
    """
    _LSTSQ_LABEL = "||AX - B||"

    def __init__(self, dims, structures, form, options=None):
        """Store keyword arguments for scipy.linalg.lstsq().

        # TODO
        Parameters
        ----------
        structure : (d,) ndarray
            Describes structure for each operator 'cAHB'
        options : dict
            Keyword arguments for scipy.linalg.lstsq().
        """
        print('WARNING - using BlockSolver')
        self.dims = dims  # [rf, rs]
        self.structures = structures  # [[True, True, True, True, False], ...]
        self.form = form  # cAH
        self.rs = None
        self.options = {} if options is None else options

    def _check_blocks(self):
        if len(self.dims) != len(self.structures):
            raise ValueError("len(dims) != len(structure)")
        for i, structure in enumerate(self.structures):
            if len(structure) != len(self.rs):
                raise ValueError(f"len(structure[{i}]) != len(rs) ({len(structure):d} != {len(self.rs):d})")
    
    def _check_is_trained(self):
        if self.rs is None:
            raise ValueError("self.rs is None - call fit() to train the lstsq solver")

    # Main routines ----------------------------------------------------------
    def fit(self, A, B):
        """Store (total) A and B."""
        _BaseSolver.fit(self, A, B)

        # Construct list of dimensions
        rs = []
        if 'c' in self.form:
            rs.append(1)
        if 'A' in self.form:
            for r in self.dims:
                rs.append(r)
        if 'H' in self.form:
            for r in self.dims:
                rs.append(r*(r+1)//2)
        print('Block dimensions [1, rf, rs, rf*(rf+1)//2, rs*(rs+1)//2] = ')
        print(rs)
        self.rs = rs

        self._check_blocks()

        return self

    def predict(self):
        """Solve the least-squares problem."""
        self._check_is_trained()

        # Allocate space for the solution.
        X = np.zeros((self.d, self.r))

        # Solve each decoupled block problem
        r0 = 0
        for i, (r, structure) in enumerate(zip(self.dims, self.structures)):

            # Construct Ai
            Ai = []
            r0j = 0
            for j, include_block in enumerate(structure):
                if include_block:
                    Ai.append(self.A[:, r0j:r0j+self.rs[j]])
                r0j += self.rs[j]
            Ai = np.hstack(Ai)

            # Construct Bi
            Bi = self.B[:, r0:r0+r]

            # Solve least squares problem
            Xi = la.lstsq(Ai, Bi, **self.options)[0]
            X[:Ai.shape[1], r0:r0+r] = Xi

            print(f"Least squares subproblem i={i:d}:")
            print(Ai.shape)
            print(Bi.shape)
            print(Xi.shape)

            # Increment forward main dimension by one physics regime
            r0 += r

        print('X.shape = ')
        print(X.shape)
        return X


# BLOCK ----------------------------------------------------------------------
class BlockTikhonovSolver(_BaseTikhonovSolver):
    pass
