# coding: utf-8
import ctypes
import glob
import hashlib
import os
import site
import sys
import warnings
from ctypes.util import find_library
from typing import ClassVar, Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning


class PyPardisoSolver:
    """
    Python interface to the Intel MKL PARDISO library for solving large sparse linear systems of equations Ax=b.

    Pardiso documentation: https://software.intel.com/en-us/node/470282

    --- Basic usage ---
    matrix type: real (float64) and nonsymetric
    methods: solve, factorize

    - use the "solve(A,b)" method to solve Ax=b for x, where A is a sparse CSR (or CSC) matrix and b is a numpy array
    - use the "factorize(A)" method first, if you intend to solve the system more than once for different right-hand
      sides, the factorization will be reused automatically afterwards


    --- Advanced usage ---
    methods: get_iparm, get_iparms, set_iparm, set_matrix_type, set_phase

    - additional options can be accessed by setting the iparms (see Pardiso documentation for description)
    - other matrix types can be chosen with the "set_matrix_type" method. complex matrix types are currently not
      supported. pypardiso is only teste for mtype=11 (real and nonsymetric)
    - the solving phases can be set with the "set_phase" method
    - The out-of-core (OOC) solver either fails or crashes my computer, be careful with iparm[60]


    --- Statistical info ---
    methods: set_statistical_info_on, set_statistical_info_off

    - the Pardiso solver writes statistical info to the C stdout if desired
    - if you use pypardiso from within a jupyter notebook you can turn the statistical info on and capture the output
      real-time by wrapping your call to "solve" with wurlitzer.sys_pipes() (https://github.com/minrk/wurlitzer,
      https://pypi.python.org/pypi/wurlitzer/)
    - wurlitzer dosen't work on windows, info appears in notebook server console window if used from jupyter notebook


    --- Memory usage ---
    methods: remove_stored_factorization, free_memory

    - remove_stored_factorization can be used to delete the wrapper's copy of matrix A
    - free_memory releases the internal memory of the solver

    """

    def __init__(self, mtype: int = 11, phase: int = 13, size_limit_storage: int = 5e7, use_64bit: bool = False):

        self.libmkl = None

        # custom mkl_rt path in environment variable
        mkl_rt = os.environ.get('PYPARDISO_MKL_RT')

        # Look for the mkl_rt shared library with ctypes.util.find_library
        if mkl_rt is None:
            mkl_rt = find_library('mkl_rt')
        # also look for mkl_rt.1, Windows-specific, see
        # https://github.com/haasad/PyPardisoProject/issues/12
        if mkl_rt is None:
            mkl_rt = find_library('mkl_rt.1')

        # If we can't find mkl_rt with find_library, we search the directory
        # tree, using a few assumptions:
        # - the shared library can be found in a subdirectory of sys.prefix
        #   https://docs.python.org/3.9/library/sys.html#sys.prefix
        #   or in the user site in case of user-local installation like
        #   `pip install --user`
        #   https://peps.python.org/pep-0370/
        #   https://docs.python.org/3/library/site.html#site.USER_BASE
        # - either in `lib` (linux and macOS) or `Library\bin` (windows)
        # - if there are multiple matches for `mkl_rt`, try shorter paths
        #   first
        if mkl_rt is None:
            globs = glob.glob(
                f'{sys.prefix}/[Ll]ib*/**/*mkl_rt*', recursive=True
            ) or glob.glob(
                f'{site.USER_BASE}/[Ll]ib*/**/*mkl_rt*', recursive=True
            ) or glob.glob(
                '/opt/intel/oneapi/mkl/**/*mkl_rt*', recursive=True
            )
            for path in sorted(globs, key=len):
                try:
                    self.libmkl = ctypes.CDLL(path)
                    print(f"Using mkl library '{path}'")
                    break
                except (OSError, ImportError):
                    pass

            if self.libmkl is None:
                raise ImportError(
                    'Shared library mkl_rt not found. '
                    'Use environment variable PYPARDISO_MKL_RT to provide a custom path.'
                )
        else:
            self.libmkl = ctypes.CDLL(mkl_rt)

        if use_64bit:
            self._mkl_pardiso = self.libmkl.pardiso_64
            c_int = ctypes.c_int64
            n_int = np.int64
        else:
            self._mkl_pardiso = self.libmkl.pardiso
            c_int = ctypes.c_int32
            n_int = np.int32

        # determine 32bit or 64bit architecture
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self._pt_type = (ctypes.c_int64, np.int64)
        else:
            self._pt_type = (ctypes.c_int32, np.int32)

        self._mkl_pardiso.argtypes = [ctypes.POINTER(self._pt_type[0]),    # pt
                                      ctypes.POINTER(c_int),               # maxfct
                                      ctypes.POINTER(c_int),               # mnum
                                      ctypes.POINTER(c_int),               # mtype
                                      ctypes.POINTER(c_int),               # phase
                                      ctypes.POINTER(c_int),               # n
                                      ctypes.POINTER(None),                # a
                                      ctypes.POINTER(c_int),               # ia
                                      ctypes.POINTER(c_int),               # ja
                                      ctypes.POINTER(c_int),               # perm
                                      ctypes.POINTER(c_int),               # nrhs
                                      ctypes.POINTER(c_int),               # iparm
                                      ctypes.POINTER(c_int),               # msglvl
                                      ctypes.POINTER(None),                # b
                                      ctypes.POINTER(None),                # x
                                      ctypes.POINTER(c_int)]               # error

        self._mkl_pardiso.restype = None

        self.pt = np.zeros(64, dtype=self._pt_type[1])
        self.iparm = np.zeros(64, dtype=n_int)
        self.perm = np.zeros(0, dtype=n_int)

        self.mtype = mtype
        self.phase = phase
        self.msglvl = False

        self.factorized_A = sp.csr_matrix((0, 0))
        self.size_limit_storage = size_limit_storage
        self._solve_transposed = False

        self.use_64bit = use_64bit

    def reconfigure(self, use_64bit: Optional[bool], msglvl: Optional[bool] = None):
        if use_64bit is not None:
            self.__init__(use_64bit=use_64bit)
        if msglvl is not None:
            self.msglvl = msglvl

    def factorize(self, A):
        """
        Factorize the matrix A, the factorization will automatically be used if the same matrix A is passed to the
        solve method. This will drastically increase the speed of solve, if solve is called more than once for the
        same matrix A

        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix), CSC matrix also possible
        """

        self._check_A(A)

        if A.nnz > self.size_limit_storage:
            self.factorized_A = self._hash_csr_matrix(A)
        else:
            self.factorized_A = A.copy()

        self.set_phase(12)
        b = np.zeros((A.shape[0], 1))
        self._call_pardiso(A, b)

    def solve(self, A, b):
        """
        solve Ax=b for x

        --- Parameters ---
        A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix), CSC matrix also possible
        b: numpy ndarray
           right-hand side(s), b.shape[0] needs to be the same as A.shape[0]

        --- Returns ---
        x: numpy ndarray
           solution of the system of linear equations, same shape as input b
        """

        self._check_A(A)
        b = self._check_b(A, b)

        if self._is_already_factorized(A):
            self.set_phase(33)
        else:
            self.set_phase(13)

        x = self._call_pardiso(A, b)

        # it is possible to call the solver with empty columns, but computationally expensive to check this
        # beforehand, therefore only the result is checked for infinite elements.
        # if not np.isfinite(x).all():
        #    warnings.warn('The result contains infinite elements. Make sure that matrix A contains no empty columns.',
        #                  PyPardisoWarning)
        # --> this check doesn't work consistently, maybe add an advanced input check method for A

        return x

    def _is_already_factorized(self, A):
        if type(self.factorized_A) == str:
            return self._hash_csr_matrix(A) == self.factorized_A
        else:
            return self._csr_matrix_equal(A, self.factorized_A)

    def _csr_matrix_equal(self, a1, a2):
        return all((np.array_equal(a1.indptr, a2.indptr),
                    np.array_equal(a1.indices, a2.indices),
                    np.array_equal(a1.data, a2.data)))

    def _hash_csr_matrix(self, matrix):
        return (hashlib.sha1(matrix.indices).hexdigest() +
                hashlib.sha1(matrix.indptr).hexdigest() +
                hashlib.sha1(matrix.data).hexdigest())

    def _check_A(self, A):
        if A.shape[0] != A.shape[1]:
            raise ValueError('Matrix A needs to be square, but has shape: {}'.format(A.shape))

        if sp.isspmatrix_csr(A):
            self._solve_transposed = False
            self.set_iparm(12, 0)
        elif sp.isspmatrix_csc(A):
            self._solve_transposed = True
            self.set_iparm(12, 1)
        else:
            msg = 'PyPardiso requires matrix A to be in CSR or CSC format, but matrix A is: {}'.format(type(A))
            raise TypeError(msg)

        # scipy allows unsorted csr-indices, which lead to completely wrong pardiso results
        if not A.has_sorted_indices:
            A.sort_indices()

        # scipy allows csr matrices with empty rows. a square matrix with an empty row is singular. calling
        # pardiso with a matrix A that contains empty rows leads to a segfault, same applies for csc with
        # empty columns
        if not np.diff(A.indptr).all():
            row_col = 'column' if self._solve_transposed else 'row'
            raise ValueError('Matrix A is singular, because it contains empty {}(s)'.format(row_col))

        if A.dtype != np.float64:
            raise TypeError('PyPardiso currently only supports float64, but matrix A has dtype: {}'.format(A.dtype))

    def _check_b(self, A, b):
        if sp.isspmatrix(b):
            warnings.warn('PyPardiso requires the right-hand side b to be a dense array for maximum efficiency',
                          SparseEfficiencyWarning)
            b = b.todense()

        # pardiso expects fortran (column-major) order for b
        if not b.flags.f_contiguous:
            b = np.asfortranarray(b)

        if b.shape[0] != A.shape[0]:
            raise ValueError("Dimension mismatch: Matrix A {} and array b {}".format(A.shape, b.shape))

        if b.dtype != np.float64:
            if b.dtype in [np.float16, np.float32, np.int16, np.int32, np.int64]:
                warnings.warn("Array b's data type was converted from {} to float64".format(str(b.dtype)),
                              PyPardisoWarning)
                b = b.astype(np.float64)
            else:
                raise TypeError('Dtype {} for array b is not supported'.format(str(b.dtype)))

        return b

    def _call_pardiso(self, A, b):

        x = np.zeros_like(b)
        if self.use_64bit:
            pardiso_error = ctypes.c_int64(0)
            c_int = ctypes.c_int64
            c_int_p = ctypes.POINTER(ctypes.c_int64)
        else:
            pardiso_error = ctypes.c_int32(0)
            c_int = ctypes.c_int32
            c_int_p = ctypes.POINTER(ctypes.c_int32)
        c_float64_p = ctypes.POINTER(ctypes.c_double)

        # 1-based indexing
        ia = A.indptr + 1
        ja = A.indices + 1

        self._mkl_pardiso(self.pt.ctypes.data_as(ctypes.POINTER(self._pt_type[0])),  # pt
                          ctypes.byref(c_int(1)),  # maxfct
                          ctypes.byref(c_int(1)),  # mnum
                          ctypes.byref(c_int(self.mtype)),  # mtype -> 11 for real-nonsymetric
                          ctypes.byref(c_int(self.phase)),  # phase -> 13
                          ctypes.byref(c_int(A.shape[0])),  # N -> number of equations/size of matrix
                          A.data.ctypes.data_as(c_float64_p),  # A -> non-zero entries in matrix
                          ia.ctypes.data_as(c_int_p),  # ia -> csr-indptr
                          ja.ctypes.data_as(c_int_p),  # ja -> csr-indices
                          self.perm.ctypes.data_as(c_int_p),  # perm -> empty
                          ctypes.byref(c_int(1 if b.ndim == 1 else b.shape[1])),  # nrhs
                          self.iparm.ctypes.data_as(c_int_p),  # iparm-array
                          ctypes.byref(c_int(self.msglvl)),  # msg-level -> 1: statistical info is printed
                          b.ctypes.data_as(c_float64_p),  # b -> right-hand side vector/matrix
                          x.ctypes.data_as(c_float64_p),  # x -> output
                          ctypes.byref(pardiso_error))  # pardiso error

        if pardiso_error.value != 0:
            raise PyPardisoError(pardiso_error.value)
        else:
            return np.ascontiguousarray(x)  # change memory-layout back from fortran to c order

    def get_iparms(self):
        """Returns a dictionary of iparms"""
        return dict(enumerate(self.iparm, 1))

    def get_iparm(self, i):
        """Returns the i-th iparm (1-based indexing)"""
        return self.iparm[i-1]

    def set_iparm(self, i, value):
        """set the i-th iparm to 'value' (1-based indexing)"""
        if i not in {1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 18, 19, 21, 24, 25, 27, 28, 31, 34, 35, 36, 37, 56, 60}:
            warnings.warn('{} is no input iparm. See the Pardiso documentation.'.format(value), PyPardisoWarning)
        self.iparm[i-1] = value

    def set_matrix_type(self, mtype):
        """Set the matrix type (see Pardiso documentation)"""
        self.mtype = mtype

    def set_statistical_info_on(self):
        """Display statistical info (appears in notebook server console window if pypardiso is
        used from jupyter notebook, use wurlitzer to redirect info to the notebook)"""
        self.msglvl = 1

    def set_statistical_info_off(self):
        """Turns statistical info off"""
        self.msglvl = 0

    def set_phase(self, phase):
        """Set the phase(s) for the solver. See the Pardiso documentation for details."""
        self.phase = phase

    def remove_stored_factorization(self):
        """removes the stored factorization, this will free the memory in python, but the factorization in pardiso
        is still accessible with a direct call to self._call_pardiso(A,b) with phase=33"""
        self.factorized_A = sp.csr_matrix((0, 0))

    def free_memory(self, everything=False):
        """release mkl's internal memory, either only for the factorization (ie the LU-decomposition) or all of
        mkl's internal memory if everything=True"""
        self.remove_stored_factorization()
        A = sp.csr_matrix((0, 0))
        b = np.zeros(0)
        self.set_phase(-1 if everything else 0)
        self._call_pardiso(A, b)
        self.set_phase(13)


class PyPardisoWarning(UserWarning):
    pass


class PyPardisoError(Exception):

    Errors: ClassVar[Dict[str, str]] = {
        "-1": "Input inconsistent.",
        "-2": "Not enough memory.",
        "-3": "Reordering problem.",
        "-4": "Zero pivot, numerical fact. or iterative refinement problem.",
        "-5": "Unclassified (internal) error.",
        "-6": "Preordering failed (matrix types 11, 13 only).",
        "-7": "Diagonal matrix problem.",
        "-8": "32-bit integer overflow problem.",
        "-10": "No license file pardiso.lic found.",
        "-11": "License is expired.",
        "-12": "Wrong username or hostname.",
        "-100": "Reached maximum number of Krylov-subspace iteration in iterative solver.",
        "-101": "No sufficient convergence in Krylov-subspace iteration within 25 iterations.",
        "-102": "Error in Krylov-subspace iteration.",
        "-103": "Break-Down in Krylov-subspace iteration.",
    }

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return (f"The Pardiso solver failed with error code {self.value}: '{self.error}'\n"
                "See Pardiso documentation for details.")

    @property
    def error(self) -> str:
        str_ = self.Errors.get(str(self.value))
        if str_ is None:
            return "Unidentified error code."
        else:
            return str_