from __future__ import print_function
from utils_LDS_integral import compute_x_T
# import numpy as np
from numpy import matlib
from scipy.linalg import expm

n = 2
k = 10
A = -matlib.rand(n, n)
x0 = 1e3*matlib.rand(n).T
a = matlib.zeros(n).T
T = 1e-2

for i in range(n):
    A[i, i] *= k
print("A\n", A)

x = compute_x_T(A, a, x0, T)
print('x(0)=         ', x0.T)
print('x=            ', x.T)

# compute approximate x
x_a = matlib.zeros(n).T
for i in range(n):
    Ai = A[i:i+1, i:i+1]
    x_a[i, 0] = expm(T*Ai)*x0[i, 0]
print('approximate x=', x_a.T)

a = A*x0
for i in range(n):
    Ai = A[i:i+1, i:i+1]
    ai = a[i, 0] - Ai*x0[i, 0]
    xi = compute_x_T(Ai, ai, x0[i, 0], T)
    x_a[i, 0] = xi[0, 0]
print('approximate x=', x_a.T)
