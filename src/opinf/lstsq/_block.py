# lstsq/_block.py
"""Operator Inference least-squares solvers for block-structured systems."""

__all__ = [
    "BlockPlainSolver",
    "BlockTikhonovSolver",
]

import abc
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

from ._base import _BaseSolver, lstsq_size
from ._tikhonov import _BaseTikhonovSolver
from ..utils import kron2c

class BlockPlainSolver(_BaseSolver):
    """Solve the l2-norm ordinary least-squares problem without any
    regularization, i.e.,

        min_{X} ||AX - B||_F^2.

    The solution is calculated using scipy.linalg.lstsq().
    """
    _LSTSQ_LABEL = "||AX - B||"

    def __init__(self, dimensions, structures, form, options=None):
        """Store keyword arguments for scipy.linalg.lstsq().

        Parameters
        ----------
        dimensions : list
            Contains dimension of state vector for each physics regime
        structure : list
            Contains operator structure for each physics regime
        options : dict
            Keyword arguments for scipy.linalg.lstsq().

        Returns
        -------
        """
        print('WARNING - using BlockSolver')
        self.dimensions = dimensions  # [rf, rs]
        self.structures = structures  # [[False, True, True, False, False, False], [...]]
        self.form = form  # 'cAHB'
        self.options = {} if options is None else options

    def _check_blocks(self):
        if len(self.dimensions) != len(self.structures):
            raise ValueError("len(dimensions) != len(structure)")
    
    def _check_is_trained(self):
        if self.dimensions is None:
            raise ValueError("self.dimensions is None - call fit() to train the lstsq solver")
        if self.structures is None:
            raise ValueError("self.structures is None - call fit() to train the lstsq solver")

    # Main routines ----------------------------------------------------------
    def fit(self, A, B):
        """Store (total) A and B."""

        # Get dimensions
        self.rf = self.dimensions[0]
        self.rs = self.dimensions[1]
        r = np.array(self.dimensions).sum()
        self.sf = self.rf*(self.rf+1)//2
        self.ss = self.rs*(self.rs+1)//2
        self.s = r*(r+1)//2

        # Construct list of dimensions for data matrix blocks
        self.dimList = []
        if 'c' in self.form:
            self.dimList.append(1)
        if 'A' in self.form:
            self.dimList.append(self.rf)
            self.dimList.append(self.rs)
        if 'H' in self.form:
            self.dimList.append(self.sf)
            self.dimList.append(self.rf*self.rs)
            self.dimList.append(self.ss)
            # A = block_structure(A, self.rf, self.rs, self.form)  # FIXME address restructuring of H blocks

        _BaseSolver.fit(self, A, B)

        self._check_blocks()

        if r != self.r:
            raise ValueError("r != self.r")

        return self

    def predict(self):
        """Solve the least-squares problem."""
        self._check_is_trained()

        X = np.zeros((self.d, self.r))

        # Solve each decoupled block problem
        r0i = 0
        for i, (ri, structure) in enumerate(zip(self.dimensions, self.structures)):
            print(f"Least squares subproblem i={i:d}:")

            # Construct Ai
            Ai = []
            r0j = 0
            for j, block in enumerate(structure):
                rj = self.dimList[j]
                if block:
                    Aij = self.A[:, r0j:r0j+rj]
                    Ai.append(Aij)
                    print(f'A{i+1:d}{j+1:d}.shape = {Aij.shape}')
                r0j += rj
            Ai = np.hstack(Ai)
            print(f'A{i+1:d}.shape = {Ai.shape}')
            print(Ai)

            # Construct Bi
            Bi = self.B[:, r0i:r0i+ri]
            print(f'B{i+1:d}.shape = {Bi.shape}')
            print(Bi)

            # Solve least squares problem
            Xi = la.lstsq(Ai, Bi, **self.options)[0]
            print(f'X{i+1:d}.shape = {Xi.shape}')

            # Insert learned operators back into main O^T operator matrix
            # X[:Ai.shape[1], r0i:r0i+ri] = Xi
            r0j_Xi = 0
            r0j = 0
            for j, block in enumerate(structure):
                rj = self.dimList[j]
                if block:
                    Xij = Xi[r0j_Xi:r0j_Xi+rj, :]
                    print(f'X{i+1:d}{j+1:d}.shape = {Xij.shape}')
                    X[r0j:r0j+rj, r0i:r0i+ri] = Xij
                    r0j_Xi += rj
                r0j += rj

            # Increment forward main dimension by one physics regime
            r0i += ri

        print('X.shape = ')
        print(X.shape)
        return X

class BlockTikhonovSolver(_BaseTikhonovSolver):
    pass

def block_structure(A, rf, rs, form):

    # Only modify quadratic portion
    j = 0
    if 'c' in form:
        j += 1
    if 'A' in form:
        j += rf + rs

    A = A.T
    Q2 = A[j:, :]

    # Get number of timesteps
    k = Q2.shape[1]

    # Create linear masks
    qf_temp = np.ones((rf, k))
    qs_temp = np.zeros((rs, k))
    q_temp = np.vstack((qf_temp, qs_temp))
    qf_mask = np.array(q_temp, dtype=bool)

    qf_temp = np.zeros((rf, k))
    qs_temp = np.ones((rs, k))
    q_temp = np.vstack((qf_temp, qs_temp))
    qs_mask = np.array(q_temp, dtype=bool)

    # Create kronecker masks
    qf2_mask = kron2c(qf_mask)
    qs2_mask = kron2c(qs_mask)
    qfs_mask = ~qf2_mask^qs2_mask

    # Construct reformatted matrix
    block_f = np.where(qf2_mask, Q2, np.nan)
    block_f = Q2[~np.isnan(block_f).any(axis=1), :]
    block_fs = np.where(qfs_mask, Q2, np.nan)
    block_fs = Q2[~np.isnan(block_fs).any(axis=1), :]
    block_s = np.where(qs2_mask, Q2, np.nan)
    block_s = Q2[~np.isnan(block_s).any(axis=1), :]
    Q2_block = np.vstack((block_f, block_fs, block_s))

    # Merge with original data matrix
    A[j:, :] = Q2_block

    return A.T
