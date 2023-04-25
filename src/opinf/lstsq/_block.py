# lstsq/_bloc.py
"""Operator Inference least-squares solvers for block-structured systems."""

__all__ = [
    "BlockPlainSolver"
    # "BlockTikhonovSolver"
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
# class BlockTikhonovSolver(_BaseTikhonovSolver):
#     """Solve the l2-norm ordinary least-squares problem with Tikhonov
#     regularization:

#         sum_{i} min_{x_i} ||Ax_i - b_i||_2^2 + ||Px_i||_2^2,    P > 0 (SPD).

#     or, written in the Frobenius norm,

#         min_{X} ||AX - B||_F^2 + ||PX||_F^2,                    P > 0 (SPD).

#     The problem is solved by taking the singular value decomposition of the
#     augmented data matrix [A.T | P.T].T, which is equivalent to solving

#         min_{X} || [A]    _  [B] ||^{2}
#                 || [P] X     [0] ||_{F}

#     or the Normal equations

#         (A.T A + P.T P) X = A.T B.
#     """
#     _LSTSQ_LABEL = r"min_{X} ||AX - B||_F^2 + ||PX||_F^2"

#     def __init__(self, regularizer, method="svd"):
#         """Store the regularizer and initialize attributes.

#         Parameters
#         ----------
#         regularizer : (d, d) or (d,) ndarray
#             Symmetric semi-positive-definite regularization matrix P
#             or, if P is diagonal, just the diagonal entries.
#         method : str
#             The strategy for solving the regularized least-squares problem.
#             * "svd": take the SVD of the stacked data matrix [A.T | P.T].T.
#             * "normal": solve the normal equations (A.T A + P.T P) X = A.T B.
#         """
#         _BaseTikhonovSolver.__init__(self)
#         self.regularizer = regularizer
#         self.method = method

#     # Properties --------------------------------------------------------------
#     def _check_regularizer_shape(self):
#         if self.regularizer.shape != (self.d, self.d):
#             raise ValueError("regularizer.shape != (d, d) (d = A.shape[1])")

#     @property
#     def regularizer(self):
#         """(d, d) ndarray:
#         symmetric semi-positive-definite regularization matrix P.
#         """
#         return self.__reg

#     @regularizer.setter
#     def regularizer(self, P):
#         """Set the regularization matrix."""
#         if sparse.issparse(P):
#             P = P.toarray()
#         elif not isinstance(P, np.ndarray):
#             P = np.array(P)

#         if P.ndim == 1:
#             if np.any(P < 0):
#                 raise ValueError("diagonal regularizer must be "
#                                  "positive semi-definite")
#             P = np.diag(P)

#         self.__reg = P
#         if self.d is not None:
#             self._check_regularizer_shape()

#     @property
#     def method(self):
#         """str : Strategy for solving the regularized least-squares problem.
#         * "svd": take the SVD of the stacked data matrix [A.T | P.T].T.
#         * "normal": solve the normal equations (A.T A + P.T P) X = A.T B.
#         """
#         return self.__method

#     @method.setter
#     def method(self, method):
#         """Set the method and precompute stuff as needed."""
#         if method not in ("svd", "normal"):
#             raise ValueError("method must be 'svd' or 'normal'")
#         self.__method = method

#     # Main routines -----------------------------------------------------------
#     def fit(self, A, B):
#         """Store A and B."""
#         _BaseTikhonovSolver.fit(self, A, B)
#         self._check_regularizer_shape()

#         # Pad B for "svd" solve.
#         self._Bpad = np.vstack((self.B, np.zeros((self.d, self.r))))

#         # Precompute Normal equations terms for "normal" solve.
#         self._AtA = self.A.T @ self.A
#         self._rhs = self.A.T @ self.B

#         return self

#     def predict(self):
#         """Solve the least-squares problem.

#         Returns
#         -------
#         X : (d, r) or (d,) ndarray
#             Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
#             solution to the subproblem with the corresponding column of B.
#             The result is flattened to a one-dimensional array if r = 1.
#         """
#         self._check_is_trained("_Bpad")

#         if self.method == "svd":
#             X = la.lstsq(np.vstack((self.A, self.regularizer)), self._Bpad)[0]
#         elif self.method == "normal":
#             lhs = self._AtA + (self.regularizer.T @ self.regularizer)
#             X = la.solve(lhs, self._rhs, assume_a="pos")

#         return np.ravel(X) if self.r == 1 else X

#     # Post-processing ---------------------------------------------------------
#     def regcond(self):
#         """Compute the 2-norm condition number of the regularized data matrix.

#         Returns
#         -------
#         rc : float ≥ 0
#             cond([A.T | P.T].T).
#         """
#         self._check_is_trained()
#         return np.linalg.cond(np.vstack((self.A, self.regularizer)))

#     def residual(self, X):
#         """Calculate the residual of the regularized problem for each column of
#         B = [ b_1 | ... | b_r ], i.e., ||Ax_i - b_i||_2^2 + ||Px_i||_2^2.

#         Parameters
#         ----------
#         X : (d, r) ndarray
#             Least-squares solution X = [ x_1 | ... | x_r ]; each column is the
#             solution to the subproblem with the corresponding column of B.

#         Returns
#         -------
#         resids : (r,) ndarray or float (r = 1)
#             Residuals ||Ax_i - b_i||_2^2 + ||Px_i||_2^2, i = 1, ..., r.
#         """
#         self._check_is_trained()
#         return self.misfit(X) + np.sum((self.regularizer @ X)**2, axis=0)
