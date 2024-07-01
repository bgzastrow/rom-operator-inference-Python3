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

class _BaseBlockSolver(_BaseSolver):
    pass

class BlockPlainSolver(_BaseSolver):
    """Solve the l2-norm ordinary least-squares problem without any
    regularization, i.e.,

    The solution is calculated using scipy.linalg.lstsq().
    """
    _LSTSQ_LABEL = ""

    def __init__(self, dimensions, structures, form, verbose=False, options=None):
        """Store keyword arguments for scipy.linalg.lstsq(). Store other inputs."""
        self.dimensions = dimensions  # [rf, rs]
        self.structures = structures  # [[False, True, True, False, False, False], [...]]
        self.form = form  # 'cAHB'
        self.options = {} if options is None else options
        self.verbose = verbose

    # Main routines ----------------------------------------------------------
    def fit(self, A, B):
        """Set up least-squares problem. Store (total) A, B."""

        # Get dimensions
        rf = self.dimensions[0]
        rs = self.dimensions[1]
        sf = rf*(rf+1)//2
        ss = rs*(rs+1)//2

        # Construct list of dimensions for data matrix blocks
        dimList = []
        if 'c' in self.form:
            dimList.append(1)
        if 'A' in self.form:
            dimList.append(rf)
            dimList.append(rs)
        if 'H' in self.form:
            dimList.append(sf)
            dimList.append(rf*rs)
            dimList.append(ss)
            # A = block_structure(A, rf, rs, self.form)  # FIXME address restructuring of H blocks
        self.dimList = dimList

        _BaseSolver.fit(self, A, B)

        return self

    def predict(self):
        """Solve the least-squares problem. Return X."""
        self._check_is_trained()

        # Initialize main operator matrix (O^T)
        X = np.zeros((self.d, self.r))

        # Solve each decoupled block problem (one block column of X=O^T)
        r0j = 0  # counter of which column (physics regime) we are working with
        for j, (rj, structure) in enumerate(zip(self.dimensions, self.structures)):
            print_verbose(f"Least squares subproblem j={j:d}:", self.verbose)

            # Construct Aj
            Aj = []
            r0i = 0  # counter of which row (operator) we are working with
            for i, block in enumerate(structure):
                ri = self.dimList[i]  # dimension of this block
                
                # If included, add 1k, Q, kron(Q), ..., to data matrix
                if block:
                    Aij = self.A[:, r0i:r0i+ri]
                    Aj.append(Aij)
                    print_verbose(f'A{j+1:d}{i+1:d}.shape = {Aij.shape}', self.verbose)

                # Increment forward by one operator type
                r0i += ri

            # Combine sub-blocks into main blocks
            Aj = np.hstack(Aj)

            # Construct Bj
            Bj = self.B[:, r0j:r0j+rj]

            # Solve least squares problem (Compute Xj)
            Xj = la.lstsq(Aj, Bj, **self.options)[0]

            print_verbose(f'\tA X = B \t{Aj.shape} {Xj.shape} = {Bj.shape}', self.verbose)

            # Insert learned operators back into main operator matrix
            r0ij = 0  # counter of which row (operator) we are working with (in learned operator submatrix, Xj)
            r0i = 0  # counter of which row (operator) we are working with (in main operator matrix, X)
            for i, block in enumerate(structure):
                ri = self.dimList[i]  # dimension of this block

                # If included, insert c, A, E, H, ..., into main operator matrix
                if block:
                    Xij = Xj[r0ij:r0ij+ri, :]
                    X[r0i:r0i+ri, r0j:r0j+rj] = Xij
                    
                    # Increment forward by one operator type (in learned operator submatrix, Xj)
                    r0ij += ri
                    print_verbose(f'X{j+1:d}{i+1:d}.shape = {Xij.shape}', self.verbose)

                # Increment forward by one operator type (in main operator matrix, X)
                r0i += ri

            # Increment forward one physics regime
            r0j += rj

        print_verbose(f'X.shape = {X.shape}', self.verbose)

        return X

class BlockTikhonovSolver(_BaseSolver):
    """Solve the l2-norm ordinary least-squares problem with regularization,
    i.e.,

    The solution is calculated using scipy.linalg.lstsq().
    """
    _LSTSQ_LABEL = ""

    def __init__(self, dimensions, structures, form, regularizer, verbose=False, options=None):
        """Store keyword arguments for scipy.linalg.lstsq(). Store other inputs."""
        self.dimensions = dimensions  # [rf, rs]
        self.structures = structures  # [[False, True, True, False, False, False], [...]]
        self.form = form  # 'cAHB'
        self.regularizer = regularizer
        self.verbose = verbose
        self.options = {} if options is None else options

    # Main routines ----------------------------------------------------------
    def fit(self, A, B):
        """Set up least-squares problem. Store (total) A, B."""

        # Get dimensions
        rf = self.dimensions[0]
        rs = self.dimensions[1]
        sf = rf*(rf+1)//2
        ss = rs*(rs+1)//2

        # Construct list of dimensions for data matrix blocks
        dimList = []
        if 'c' in self.form:
            dimList.append(1)
        if 'A' in self.form:
            dimList.append(rf)
            dimList.append(rs)
        if 'H' in self.form:
            dimList.append(sf)
            dimList.append(rf*rs)
            dimList.append(ss)
            # A = block_structure(A, rf, rs, self.form)  # FIXME address restructuring of H blocks
        self.dimList = dimList

        _BaseSolver.fit(self, A, B)

        return self

    def predict(self):
        """Solve the least-squares problem. Return X."""
        self._check_is_trained()

        # Initialize main operator matrix (O^T)
        X = np.zeros((self.d, self.r))

        # Solve each decoupled block problem (one block column of X=O^T)
        r0j = 0  # counter of which column (physics regime) we are working with
        for j, (rj, structure) in enumerate(zip(self.dimensions, self.structures)):
            print_verbose(f"Least squares subproblem j={j:d}:", self.verbose)

            # Construct Aj, Pj
            Aj, Pj = [], []
            r0i = 0  # counter of which row (operator) we are working with
            for i, block in enumerate(structure):
                ri = self.dimList[i]  # dimension of this block
                
                # If included, add 1k, Q, kron(Q), ..., to data matrix
                if block:
                    Aij = self.A[:, r0i:r0i+ri]
                    Aj.append(Aij)
                    print_verbose(f'\tA{i:d}.shape = {Aij.shape}', self.verbose)

                    Pij = self.regularizer[r0i:r0i+ri, r0i:r0i+ri]
                    Pj.append(Pij)
                    print_verbose(f'\tP{i:d}.shape = {Pij.shape}', self.verbose)

                # Increment forward by one operator type
                r0i += ri

            # Combine sub-blocks into main blocks
            Aj = np.hstack(Aj)
            Pj = la.block_diag(*Pj)

            # Construct Bj
            Bj = self.B[:, r0j:r0j+rj]

            # Prepare for SVD solve
            Atilde = np.vstack((Aj, Pj))
            Btilde = np.vstack((Bj, np.zeros((Pj.shape[0], Bj.shape[1]))))

            # Solve least squares problem (Compute Xj)
            Xj = la.lstsq(Atilde, Btilde, **self.options)[0]

            print_verbose(f'\tA X = B \t{Aj.shape} {Xj.shape} = {Bj.shape}', self.verbose)
            print_verbose(f'\tP \t\t{Pj.shape}', self.verbose)

            # Insert learned operators back into main operator matrix
            r0ij = 0  # counter of which row (operator) we are working with (in learned operator submatrix, Xj)
            r0i = 0  # counter of which row (operator) we are working with (in main operator matrix, X)
            for i, block in enumerate(structure):
                ri = self.dimList[i]  # dimension of this block

                # If included, insert c, A, E, H, ..., into main operator matrix
                if block:
                    Xij = Xj[r0ij:r0ij+ri, :]
                    X[r0i:r0i+ri, r0j:r0j+rj] = Xij
                    
                    # Increment forward by one operator type (in learned operator submatrix, Xj)
                    r0ij += ri
                    print_verbose(f'X{j+1:d}{i+1:d}.shape = {Xij.shape}', self.verbose)

                # Increment forward by one operator type (in main operator matrix, X)
                r0i += ri

            # Increment forward one physics regime
            r0j += rj

        print_verbose(f'X.shape = {X.shape}', self.verbose)

        return X

def print_verbose(string, status):
    if status:
        print(string)

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
