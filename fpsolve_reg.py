"""
Package to solve Fokker-Planck equations

* Steady-state Fourier-space solver for the PDF
* Adjoint finite difference solver for first/second moments

(Only 1D and 2D implemented so far)

Jared Callaham (2020)
"""

import numpy as np
from numpy.fft import fft, fftn, fftfreq, ifftn
from scipy import linalg, sparse, integrate, special
import itertools as it
import time as t
from multiprocessing import Pool
from functools import partial
from functions_regression import arg_sorted


class SteadyFP:
    """
    Solver object for steady-state Fokker-Planck equation

    Initializing this independently avoids having to re-initialize all of the indexing arrays
      for repeated loops with different drift and diffusion

    Jared Callaham (2020)
    """

    def __init__(self, N, dx):
        """
        ndim - number of dimensions
        N - array of ndim ints: grid resolution N[0] x N[1] x ... x N[ndim-1]
        dx - grid spacing (array of floats)
        """

        if isinstance(N, int):
            self.ndim = 1
        else:
            self.ndim = len(N)

        self.N = N
        self.dx = dx

        # Set up indexing matrices for ndim=1, 2
        if self.ndim == 1:
            self.k = 2*np.pi*fftfreq(N, dx)
            self.idx = np.zeros((self.N, self.N), dtype=np.int32)
            for i in range(self.N):
                self.idx[i, :] = i-np.arange(N)

        elif self.ndim == 2:
            # Fourier frequencies
            self.k = [2*np.pi*fftfreq(N[i], dx[i]) for i in range(self.ndim)]
            self.idx = np.zeros(
                (2, self.N[0], self.N[1], self.N[0], self.N[1]), dtype=np.int32)

            for m in range(N[0]):
                for n in range(N[1]):
                    self.idx[0, m, n, :, :] = m - \
                        np.tile(np.arange(N[0]), [N[1], 1]).T
                    self.idx[1, m, n, :, :] = n - \
                        np.tile(np.arange(N[1]), [N[0], 1])

        else:
            print("WARNING: NOT IMPLEMENTED FOR HIGHER DIMENSIONS")

        self.A = None  # Need to initialize with precompute_operator

    def precompute_operator(self, f, a):
        """
        f - array of drift coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        a - array of diffusion coefficients on domain (ndim x N[0] x N[1] x ... x N[ndim])
        NOTE: To generalize to covariate noise, would need to add a dimension to a
        """

        if self.ndim == 1:
            f_hat = self.dx*fftn(f)
            a_hat = self.dx*fftn(a)

            # Set up spectral projection operator
            self.A = np.einsum('i,ij->ij', -1j*self.k, f_hat[self.idx]) \
                + np.einsum('i,ij->ij', -self.k**2, a_hat[self.idx])

        if self.ndim == 2:
            # Initialize Fourier transformed coefficients
            f_hat = np.zeros(
                np.append([self.ndim], self.N), dtype=np.complex64)
            N_a = self.ndim*(self.ndim + 1)//2
            a_hat = np.zeros(np.append([N_a], self.N), dtype=np.complex64)

            for i in range(self.ndim):
                f_hat[i] = np.prod(self.dx)*fftn(f[i])

            for i in range(N_a):
                a_hat[i] = np.prod(self.dx)*fftn(a[i])

            self.A = -1j*np.einsum('i,ijkl->ijkl', self.k[0], f_hat[0, self.idx[0], self.idx[1]]) \
                     - 1j*np.einsum('j,ijkl->ijkl', self.k[1], f_hat[1, self.idx[0], self.idx[1]]) \
                     - np.einsum('i,ijkl->ijkl', self.k[0]**2, a_hat[0, self.idx[0], self.idx[1]]) \
                     - np.einsum('j,ijkl->ijkl', self.k[1]**2, a_hat[2, self.idx[0], self.idx[1]]) \
                     - np.einsum('i,j,ijkl->ijkl', 2
                                 * self.k[0], self.k[1], a_hat[1, self.idx[0], self.idx[1]])

            self.A = np.reshape(self.A, (np.prod(self.N), np.prod(self.N)))

    def solve(self, f, a):
        """
        Solve Fokker-Planck equation from input drift coefficients
        """
        self.precompute_operator(f, a)
        q_hat = np.linalg.lstsq(self.A[1:, 1:], -self.A[1:, 0], rcond=1e-6)[0]
        # q_hat = np.linalg.inv(self.A[1:, 1:]).dot(-self.A[1:, 0])
        q_hat = np.append([1], q_hat)
        return np.real(ifftn(np.reshape(q_hat, self.N)))/np.prod(self.dx)


class SteadyFP_reflectBC:
    """
    Solver object for steady-state Fokker-Planck equation

    Initializing this independently avoids having to re-initialize all of the indexing arrays
      for repeated loops with different drift and diffusion

    Jared Callaham (2020)
    """

    def __init__(self, N, dx):
        """
        ndim - number of dimensions
        N - array of ndim ints: grid resolution N[0] x N[1] x ... x N[ndim-1]
        dx - grid spacing (array of floats)
        """
        self.N = N
        self.dx = dx

    def solve(self, f, a):
        """
        Solve Fokker-Planck equation from input drift coefficients
        """
        with warnings.catch_warnings(record=True) as w:
            p_est = (np.exp(np.cumsum((f/a)*self.dx)))/a
            if len(w) > 0:
                print('Some warning, printing f and a:',f,a)

        with warnings.catch_warnings(record=True) as w:
            p_est = p_est/(np.sum(p_est)*self.dx)
            if len(w) > 0:
                print('Some warning, print p_est ', p_est)
                print('Some warning, print f ', f)
                print('Some warning, print a ', a)

        return p_est





class AdjFP:
    """
    Solver object for adjoint Fokker-Planck equation

    Jared Callaham (2020)
    """

    # 1D derivative operators
    @staticmethod
    def derivs1d(x):
        N = len(x)
        dx = x[1]-x[0]
        one = np.ones((N))

        # First derivative
        Dx = sparse.diags([one, -one], [1, -1], shape=(N, N))
        Dx = sparse.lil_matrix(Dx)
        # Forward/backwards difference at boundaries
        Dx[0, :3] = [-3, 4, -1]
        Dx[-1, -3:] = [1, -4, 3]
        Dx = sparse.csr_matrix(Dx)/(2*dx)

        # Second derivative
        Dxx = sparse.diags([one, -2*one, one], [1, 0, -1], shape=(N, N))
        Dxx = sparse.lil_matrix(Dxx)
        # Forwards/backwards differences  (second-order accurate)
        Dxx[-1, -4:] = [1.25, -2.75, 1.75, -.25]
        Dxx[0, :4] = [-.25, 1.75, -2.75, 1.25]
        Dxx = sparse.csr_matrix(Dxx)/(dx**2)

        return Dx, Dxx

    @staticmethod
    def derivs2d(x, y):
        hx, hy = x[1]-x[0], y[1]-y[0]
        Nx, Ny = len(x), len(y)

        Dy = sparse.diags([-1, 1], [-1, 1], shape=(Ny, Ny)).toarray()

        # Second-order forward/backwards at boundaries
        Dy[0, :3] = np.array([-3, 4, -1])
        Dy[-1, -3:] = np.array([1, -4, 3])
        # Repeat for each x-location
        Dy = linalg.block_diag(
            *Dy.reshape(1, Ny, Ny).repeat(Nx, axis=0))/(2*hy)
        Dy = sparse.csr_matrix(Dy)

        Dx = sparse.diags([-1, 1], [-Ny, Ny], shape=(Nx*Ny, Nx*Ny)).toarray()
        # Second-order forwards/backwards at boundaries
        for i in range(Ny):
            Dx[i, i] = -3
            Dx[i, Ny+i] = 4
            Dx[i, 2*Ny+i] = -1
            Dx[-(i+1), -(i+1)] = 3
            Dx[-(i+1), -(Ny+i+1)] = -4
            Dx[-(i+1), -(2*Ny+i+1)] = 1
        Dx = sparse.csr_matrix(Dx)/(2*hx)

        Dxx = sparse.csr_matrix(Dx @ Dx)
        Dyy = sparse.csr_matrix(Dy @ Dy)
        Dxy = sparse.csr_matrix(Dx @ Dy)

        return Dx, Dy, Dxx, Dyy, Dxy

    def __init__(self, x, ndim=1, **kwargs):
        """
        x - uniform grid (array of floats)
        """
        self.ndim = ndim

        if self.ndim == 1:
            self.N = [len(x)]
            self.dx = [x[1]-x[0]]
            self.x = [x]
            self.Dx, self.Dxx = AdjFP.derivs1d(x)
            self.precompute_operator = self.operator1d
        else:
            self.x = x
            self.N = [len(x[i]) for i in range(len(x))]
            self.dx = [x[i][1]-x[i][0] for i in range(len(x))]
            self.Dx, self.Dy, self.Dxx, self.Dyy, self.Dxy = AdjFP.derivs2d(*x)
            self.precompute_operator = self.operator2d

        if len(kwargs) != 0:  # Choose solver type : exp or differential equation
            solve_type = kwargs.get('solve')

            if solve_type == 'diff':
                self.solve = self.solve_PDE
            elif solve_type == 'exp':
                self.solve = self.solve_exp
            else:
                return 'ERROR'

        self.XX = np.meshgrid(*self.x, indexing='ij')
        self.precompute_moments()

    def precompute_moments(self):
        N = self.ndim
        self.m1 = np.zeros([N, np.prod(self.N), np.prod(self.N)])
        self.m2 = np.zeros([N*(N+1)//2, np.prod(self.N), np.prod(self.N)])

        for d in range(self.ndim):
            for i in range(np.prod(self.N)):
                self.m1[d, i, :] = self.XX[d].flatten() - \
                                                      self.XX[d].flatten()[i]

        list_to_combine = range(N)
        combinations = list(
            it.combinations_with_replacement(list_to_combine, 2))
        d = 0
        for dims in combinations:
            for i in range(np.prod(self.N)):
                interm = (self.XX[dims[0]].flatten() - self.XX[dims[0]].flatten()[i])*(
                    self.XX[dims[1]].flatten() - self.XX[dims[1]].flatten()[i])
                self.m2[d, i, :] = interm
            d += 1

    def operator1d(self, f, a):
        self.L = sparse.dia_matrix(sparse.diags(
            f) @ self.Dx + sparse.diags(a) @ self.Dxx)

    def operator2d(self, f, a):
        self.L = sparse.dia_matrix(sparse.diags(f[0]) @ self.Dx + sparse.diags(f[1]) @ self.Dy
                                   + sparse.diags(a[0]) @ self.Dxx
                                   + sparse.diags(a[2]) @ self.Dyy
                                   + 2*sparse.diags(a[1]) @ self.Dxy)

    def solve_exp(self, tau):
        if self.L is None:
            print("Need to initialize operator")
            return None

        N = self.ndim

        L_tau = linalg.expm(self.L.todense()*tau)

        f_tau = []
        a_tau = []

        for d in range(N):
            f_tau.append(np.einsum('ij,ij->i', L_tau, self.m1[d])/tau)

        for d in range(N*(N+1)//2):
            a_tau.append(np.einsum('ij,ij->i', L_tau, self.m2[d])/(2*tau))

        return f_tau, a_tau

    def solve_PDE(self, tau, **kwargs):
        if self.L is None:
            print("Need to initialize operator")
            return None

        Nx = np.prod(np.array(self.N))

        X = []
        w0 = []

        if self.ndim > 1:  # Build Vectors from meshgrid
            sizes = self.XX[0].shape
            stack = np.stack(self.XX, axis=-1)
            X = np.reshape(stack, (np.prod(sizes), 2)).T
        else:  # Not necessesary for 1D
            X = np.squeeze(np.array(self.x))

        products = list(it.product(range(3), repeat=self.ndim))

        for coeffs in products:  # Build w0 from powers of X
            coeff_arr = np.array(coeffs)
            if np.sum(coeff_arr) <= 2 and not np.all(coeff_arr == 0):
                if self.ndim == 1:
                    w0.append(X**coeff_arr)
                else:
                    w0.append(np.prod(X.T**coeff_arr, axis=-1).flatten())

        w_tau = []
        Num_equ = self.ndim + self.ndim*(self.ndim + 1)//2
        # Neq for f_tau and Neq(Neq + 1)//2 for a_tau

        b = t.time()
        if len(kwargs) != 0:  # Multiprocessing over solving different ODE
            with Pool(processes=kwargs["Nbr_proc"]) as pool:
                w_tau = pool.map(partial(work_PDE, self.L.todense(), tau), w0)
            w_tau = [np.ones((Nx,))] + w_tau

        else:  # loop for over solving different ODE
            w_tau.append(np.ones((Nx,)))
            for i in range(Num_equ):
                w_tau.append(work_PDE(self.L.todense(), tau, w0[i]))

        print(t.time() - b)
        t.sleep(100)

        ## Computing KM coefficients
        f_tau = []
        a_tau = []
        p_f = []
        p_a = []

        for coeffs in products:
            a = []
            coeffs_i = np.array(coeffs)

            if np.sum(coeffs_i) <= 2 and np.sum(coeffs_i) != 0:
                for coeffs_ in products:
                    coeffs_j = np.array(coeffs_)
                    if np.sum(coeffs_j) <= 2:
                        if self.ndim > 1:  # See associated formula.
                            a.append(np.prod((-X.T)**(coeffs_i - coeffs_j)
                                             * special.binom(coeffs_i, coeffs_j), axis=-1))

                        else:
                            a.append((-X)**(coeffs_i - coeffs_j)
                                     * special.binom(coeffs_i, coeffs_j))

                a = np.stack(a)  # Regroup everything together in numpy

                if np.sum(coeffs_i) == 1:
                    f_tau.append(np.sum(a * w_tau, axis=0)/tau)
                    p_f.append(list(coeffs_i))
                if np.sum(coeffs_i) == 2:
                    a_tau.append(np.sum(a * w_tau, axis=0)/2/tau)
                    p_a.append(list(coeffs_i))

        # Python sort f_tau and a_tau with x first, y then, z then, etc...
        f_tau = [f_tau[i] for i in arg_sorted(p_f, reverse=True)]
        a_tau = [a_tau[i] for i in arg_sorted(p_a, reverse=True)]

        return f_tau, a_tau


def work_PDE(L, tau, w0):
    def func_diff(t, w): return L @ np.squeeze(w)
    t_span = (0, tau)
    res = integrate.solve_ivp(
        func_diff, t_span, np.squeeze(w0), method='DOP853')

    return res.y[:, -1]
    return res.y[:, -1]

    return res.y[:, -1]
