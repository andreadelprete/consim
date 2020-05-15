'''
Consider a Linear Dynamical System:
  dx(t)/dt = A * x(t) + a
Given x(0) and T I want to compute:
- x(T)
- The integral of x(t) for t in [0, T]
- The double integral of x(t) for t in [0, T]
'''
from __future__ import print_function
from scipy.sparse.linalg.matfuncs import _ExpmPadeHelper, _ell, _solve_P_Q
import numpy as np
from numpy.linalg import solve
from numpy.linalg import eigvals
from scipy.linalg import matrix_balance
# import matplotlib.pyplot as plt

np.set_printoptions(precision=2, linewidth=200, suppress=True)

class ExponentialMatrixHelper:
    
    def __init__(self):
        self.mat_mult = 0           # number of matrix-matrix multilpications used at last computation
        self.mat_mult_in_theory = 0 # theoretical number of mat-mat multiplications needed
        self.mat_norm = 1.0
        self.use_new_expm_alg = True
    
    
    def compute_mat_mult(self, A):
        use_exact_onenorm = A.shape[0] < 200
        h = _ExpmPadeHelper(A, use_exact_onenorm=use_exact_onenorm)
        
        # Try Pade order 3.
        eta_1 = max(h.d4_loose, h.d6_loose)
        if (eta_1 < 1.495585217958292e-002 and _ell(h.A, 3) == 0):
            return 2    
        # Try Pade order 5.
        eta_2 = max(h.d4_tight, h.d6_loose)
        if (eta_2 < 2.539398330063230e-001 and _ell(h.A, 5) == 0):
            return 3
        # Try Pade orders 7 and 9.
        eta_3 = max(h.d6_tight, h.d8_loose)
        if (eta_3 < 9.504178996162932e-001 and _ell(h.A, 7) == 0):
            return  4
        if (eta_3 < 2.097847961257068e+000 and _ell(h.A, 9) == 0):
            return 5
        # Use Pade order 13.
        if(self.use_new_expm_alg):
            eta_3 = max(h.d6_tight, h.d8_loose)
            eta_4 = max(h.d8_loose, h.d10_loose)
            eta_5 = min(eta_3, eta_4)
            theta_13 = 4.25
            # Choose smallest s>=0 such that 2**(-s) eta_5 <= theta_13
            if eta_5 == 0:
                # Nilpotent special case
                s = 0
            else:
                s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
            s = s + _ell(2**-s * h.A, 13)
        else:
            maxnorm = 5.371920351148152
            s = max(0, int(np.ceil(np.log2(np.linalg.norm(A,1) / maxnorm))))
        return 6+s
      
    def pade1(self, A):
        ''' 
            U = A
            V = 2
            P = V + U = 2 + A
            Q = V - U = 2 - A
        '''
        V = 2.*np.identity(A.shape[0])
        U = A
        return U, V
        
    def pade2(self, A):
        ''' 
            U = 12 + A2
            V = 6*A
            P = V + U = 12 + 6*A + A2
            Q = V - U = 12 - 6*A + A2
        '''
        V = 12.*np.identity(A.shape[0]) + A@A
        U = 6.*A
        return U, V
        
    def pade3(self, A):
        ''' 
            U = A3 + 60*A
            V = 12*A2 + 120
            P = U + V = A3 + 12*A2 + 60*A  + 120
            Q = V - U = -A3 + 12*A2 - 60*A  + 120
        '''
        b = (120., 60., 12., 1.)
        A2 = A @ A
        I = np.identity(A.shape[0])
        U = A @ (b[3]*A2 + b[1]*I)
        V = b[2]*A2 + b[0]*I
        return U, V
        
    def expm_times_v(self, A, v, max_mat_mult=100, balance=True):
        ''' max_mat_mult can be used to reduce the computational complexity of the exponential 
            6: at most a Pade order 13 can be used but with no scaling
            5: at most a Pade order 9 is used
            4: at most a Pade order 7 is used
            3: at most a Pade order 5 is used
            2: at most a Pade order 3 is used
        '''
        if(np.any(np.isnan(A))):
            print("Matrix A contains nan")
            
        # Compute expm(A)*v
        if balance:
            A_bal, D = matrix_balance(A, permute=False)
            Dinv = np.copy(D)
            for i in range(D.shape[0]):
                Dinv[i,i] = 1.0/D[i,i]
#            assert(np.max(np.abs(A_bal-(Dinv@A@D)))==0.0)
        else:
            A_bal = A
            
        # Hardcode a matrix order threshold for exact vs. estimated one-norms.
        use_exact_onenorm = A.shape[0] < 200
        h = _ExpmPadeHelper(A_bal, use_exact_onenorm=use_exact_onenorm)
        structure = None
        
        # Compute the number of mat-mat multiplications needed in theory
        self.mat_mult_in_theory = self.compute_mat_mult(A_bal)
        self.mat_mult = min(self.mat_mult_in_theory, max_mat_mult)
        self.mat_norm = np.linalg.norm(A_bal, 1)
        
        if self.mat_mult <= 0:
            U, V = self.pade1(A_bal)
            X = _solve_P_Q(U, V, structure=structure)
            
        if self.mat_mult == 1:
            U, V = self.pade2(A_bal)
            X = _solve_P_Q(U, V, structure=structure)
            
        if self.mat_mult == 2:
            U, V = h.pade3()
#            U_, V_ = self.pade3(A_bal)
#            assert(np.max(np.abs(U-U_))==0.0)
#            assert(np.max(np.abs(V-V_))==0.0)
            X = _solve_P_Q(U, V, structure=structure)
    
        # Try Pade order 5.
        if self.mat_mult == 3:
            U, V = h.pade5()
            X = _solve_P_Q(U, V, structure=structure)
    
        # Try Pade orders 7 and 9.
        if self.mat_mult == 4:
            U, V = h.pade7()
            X = _solve_P_Q(U, V, structure=structure)
            
        if self.mat_mult == 5:
            U, V = h.pade9()
            X = _solve_P_Q(U, V, structure=structure)    
        
        if self.mat_mult > 5:
            s = self.mat_mult-6
            U, V = h.pade13_scaled(s)
            X = _solve_P_Q(U, V)
            # X = r_13(A)^(2^s) by repeated squaring.
            for i in range(s):
                X = X.dot(X)
        
        if(balance):
            X = D @ X @ Dinv
#        assert(np.max(np.abs(expm(A) - X)) == 0.0)
            
        res = X.dot(v)
        return res
    
    
    def compute_x_T(self, A, a, x0, T, max_mat_mult=100, balance=True):
        n = A.shape[0]
        C = np.zeros((n+1, n+1))
        C[0:n,     0:n] = A
        C[0:n,     n] = a
        z0 = np.zeros(n+1)
        z0[:n] = x0
        z0[-1] = 1.0
#        e_TC = expm(T*C, verbose=True)
#        z = e_TC@z0
        z = self.expm_times_v(T*C, z0, max_mat_mult, balance)
        x_T = z[:n]
        return x_T
    
    
    def compute_integral_x_T(self, A, a, x0, T, max_mat_mult=100, balance=True):
        n = A.shape[0]
        C = np.zeros((n+2, n+2))
        C[0:n,     0:n] = A
        C[0:n,     n] = a
        C[0:n,     n+1] = x0
        C[n:n+1, n+1:] = 1.0
        z0 = np.zeros(n+2)
        z0[-1] = 1.0
    #    e_TC = expm(T@C, verbose=True)
    #    z = e_TC@z0
        z = self.expm_times_v(T*C, z0, max_mat_mult, balance)
        int_x = z[:n]
        return int_x
    
    
    def compute_double_integral_x_T(self, A, a, x0, T, max_mat_mult=100, balance=True):
        n = A.shape[0]
        C = np.zeros((n+3, n+3))
        C[0:n,     0:n] = A
        C[0:n,     n] = a
        C[0:n,     n+1] = x0
        C[n:n+2, n+1:] = np.eye(2)
        z0 = np.zeros(n+3)
        z0[-1] = 1.0
    #    e_TC = expm(T*C, verbose=True)
    #    z = e_TC@z0
        z = self.expm_times_v(T*C, z0, max_mat_mult, balance)
        int2_x = z[:n]
        return int2_x

    
    
def print_error(x_exact, x_approx):
    print("Approximation error: ", np.max(np.abs(x_exact-x_approx) / np.abs(x_exact)))


if __name__ == '__main__':
    import time
    from numpy.random import random as rand
    N_TESTS = 1
    T = 0.001
    dt = 1e-7
    n = 1*3*2
    n2 = int(n/2)
    stiffness = 1e5
    damping = 1e2
    x0 = rand(n)
    a = rand(n)
    U = rand((n2, n2))
    Upsilon = U@U.T
    K = np.eye(n2)*stiffness
    B = np.eye(n2)*damping
    A = np.block([[np.zeros((n2, n2)), np.eye(n2)],
                      [-Upsilon@K,      -Upsilon@B]])
#    A  = rand((n, n))
    
    helper = ExponentialMatrixHelper()

    # print("x(0) is:", x0.T)
    # print("a is:   ", a.T)
    print("State size n:", n)
    print("Eigenvalues of A:", np.sort_complex(eigvals(A)).T)
    print("")
    
    MAX_MAT_MULT = 0

    start_time = time.time()
    x_T = helper.compute_x_T(A, a, x0, T)
    time_exact = time.time()-start_time
    
    start_time = time.time()
    x_T_approx = helper.compute_x_T(A, a, x0, T, MAX_MAT_MULT)
    time_approx = time.time()-start_time
    
    print("Mat-mult needed in theory:", helper.mat_mult_in_theory)
    print("Mat-mult actually used:   ", helper.mat_mult)
    print("Approximated x(T) computed in             ", 1e3*time_approx)
    print("Exact x(T) computed in                    ", 1e3*time_exact)
    print("Approximated x(T)", x_T_approx.T)
    print("Exact x(T)       ", x_T.T)
    print_error(x_T, x_T_approx)
    print("")

    start_time = time.time()
    int_x_T = helper.compute_integral_x_T(A, a, x0, T)
    time_exact = time.time()-start_time
    
    start_time = time.time()
    int_x_T_approx = helper.compute_integral_x_T(A, a, x0, T, MAX_MAT_MULT)
    time_approx = time.time()-start_time

    print("Mat-mult needed in theory:", helper.mat_mult_in_theory)
    print("Mat-mult actually used:   ", helper.mat_mult)
    print("Approximated int x(T) computed in               ", 1e3*time_approx)
    print("Exact int x(T) computed in                      ", 1e3*time_exact)
    print_error(int_x_T, int_x_T_approx)
    print("")
