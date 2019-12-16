from __future__ import print_function
from utils_LDS_integral import compute_x_T, compute_integral_x_T
import numpy as np
from numpy import matlib
from scipy.linalg import expm

def test_random_matrix():
    n = 2
    k = 10
    A = -matlib.rand(n,n)
    x0 = 1e3*matlib.rand(n).T
    a = matlib.zeros(n).T
    T = 1e-2
    
    for i in range(n):
        A[i,i] *= k
    print("A\n", A)
    
    x = compute_x_T(A, a, x0, T)
    print('x(0)=         ', x0.T)
    print('x=            ', x.T)
    
    # compute approximate x
    x_a = matlib.zeros(n).T
    for i in range(n):
        Ai = A[i:i+1,i:i+1]
        x_a[i,0] = expm(T*Ai)*x0[i,0]
    print('approximate x=', x_a.T)
    
    a = A*x0
    for i in range(n):
        Ai = A[i:i+1,i:i+1]
        ai = a[i,0] - Ai*x0[i,0]
        xi = compute_x_T(Ai, ai, x0[i,0], T)
        x_a[i,0] = xi[0,0]
    print('approximate x=', x_a.T)
    
def test_algo_1(A, x0):
    ''' This algorithm normalizes each row by the sum of all its elements (in absolute value).
        Then it loops over the rows, taking all elements that contribute at least
        10% to the total.
        This algorithm can easily fail.
    '''
    n = A.shape[0]
    dx0 = A*x0
    x = matlib.zeros(n).T
    
    # normalize rows of Upsilon
    U = A.copy()
    for i in range(U.shape[0]):
        U[i,:] = np.abs(U[i,:])/np.sum(np.abs(U[i,:]))
        
    done = False
    left = range(n)
    print("\nU\n", U)
    while not done:
        # find elements have at least 10% effect of current item derivative (compared to sum of all elements)
        print("Analyze row", left[0], ':', U[left[0],:])
        ii = np.where(U[left[0],:].A1>0.1)[0]
        print("Select elements:", ii, "which give %.1f %% coverage"%(1e2*np.sum(U[left[0],ii])))        
        Ax = A[ii,:][:,ii]
        ax = dx0[ii]
        ax -= Ax * x0[ii]
        x0x = x0[ii]
        x[ii] = compute_x_T(Ax, ax, x0x, 1, invertible_A=False)        
        for i in ii:
            left.remove(i)  
        if len(left)==0:
            done = True
    return x
    

def test_algo_2(A, x0, thr=0.8):
    ''' This algorithm normalizes each row by the sum of all its elements (in absolute value).
        Then it loops over the rows:
        - start with diagonal element i (which should be the one contributing the most)
        - if contribution is not enough (e.g., <80%), select another element j to add
        - the best element to add is such that A(i,j) is large and also A(j,i) is large
          because otherwise i is benefiting from having j in its group, but j is not
        - a reasonable metric could be to find j that maximizes A(i,j)+A(j,i)
        - At this point i and j belong to same group, so should be treated together
        - If current contributions of elements in the group is not above given threshold
          we must find a new element to add to the group
        - the best element to add is the one contribute the most to both i and j, while benefiting
          from i and j, so it must maximize A(i,k)+A(j,k)+A(k,i)+A(k,j)
        Variant: (not implemented yet)
        - rather than maximizing the sum of the elements, we could normalize each element
          wrt how much is still needed to reach the threshold for that row
          
        Note that to have a robust algorithm you should process all rows before being sure 
        of the selected groups because there might be elements with high enough diagonal, 
        but that are needed by other elements
    '''
    n = A.shape[0]
    dx0 = A*x0
    x = matlib.zeros(n).T
    
    # normalize rows of Upsilon
    U = A.copy()
    for i in range(U.shape[0]):
        U[i,:] = np.abs(U[i,:])/np.sum(np.abs(U[i,:]))
        
    done = False
    left = range(n)
    print("\nU\n", U)
    while not done:
        i = left[0]
        print("Analyze row", i, ':', U[i,:])
        
        group = [i]
        done_row = False
        while not done_row:
            coverage = np.sum(U[i,group])
            print("Current elements", group, "give %.1f %% coverage"%(1e2*coverage))
            if coverage >= thr:
                done_row = True
                break
            
            # find new element j that contributes the most while benefiting the most from current elements
            gains = np.zeros(n)
            for j in left:
                if j in group: continue
                for ii in group:
                    gains[j] += U[ii,j] + U[j,ii]
#                print("Gain from element %d: %.2f"%(j, gains[j]))
            j = np.argmax(gains)
            print("Element with highest gain is", j)
            group += [j]
                
        Ax = A[group,:][:,group]
        ax = dx0[group]
        ax -= Ax * x0[group]
#        ax[k:] += b[ii]
        x0x = x0[group]
        x[group] = compute_x_T(Ax, ax, x0x, 1, invertible_A=False)
        
        for i in group:
            left.remove(i)
  
        if len(left)==0:
            done = True
    return x
    

def test_algo_3(A, x0, thr=0.1):
    ''' First test of iterative algorithm with convergence criterium
    '''
    n = 2
    k = 10
    A = -1e2*matlib.rand(n,n)
    x0 = 1e3*matlib.rand(n).T
    a = matlib.zeros(n).T
    T = 1e-2
    
    for i in range(n):
        A[i,i] *= k
    print("A\n", A)
    
    x = compute_x_T(A, a, x0, T)
    x_avg = compute_integral_x_T(A, a, x0, T) / T
    print('x(0)=         ', x0.T)
    print('x(T)=         ', x.T)
    print('x average=    ', x_avg.T)
    
    x_a = matlib.zeros(n).T
    a = A*x0
    for i in range(n):
        Ai = A[i:i+1,i:i+1]
        ai = a[i,0] - Ai*x0[i,0]
        xi = compute_x_T(Ai, ai, x0[i,0], T)
        x_a[i,0] = xi[0,0]
    print("\nOLD METHOD (1-SHOT)")
    print('approximate x=', x_a.T)
    
    # compute approximate x
    x_a = matlib.zeros(n).T
    a = A*x0
    a_old = matlib.zeros(n).T
    x_avg = matlib.zeros(n).T
    x_avg = x0.copy()
    i = 0; j=1
    a[0,0] = A[0,1]*x_avg[1,0]
    a[1,0] = A[1,0]*x_avg[0,0]
    Ai = A[i:i+1,i:i+1]
    x_avg[i,0] = compute_integral_x_T(Ai, a[i,0], x0[i,0], T) / T
    a[j,0] = A[j,i]*x_avg[i,0]
    x_a[i,0] = compute_x_T(Ai, a[i,0], x0[i,0], T)
    print("\nInitialization:")
    print('approximate x(T)=', x_a.T)
    print('x avg = \t', x_avg.T)
        
    count = 0
    max_count = 10
    while True:
        i = 1; j=0
        Ai = A[i:i+1,i:i+1]
        a_old[j,0] = a[j,0]
        x_avg[i,0] = compute_integral_x_T(Ai, a[i,0], x0[i,0], T) / T
        a[j,0] = A[j,i]*x_avg[i,0]
        x_a[i,0] = compute_x_T(Ai, a[i,0], x0[i,0], T)

        i = 0; j = 1
        Ai = A[i:i+1,i:i+1]
        a_old[j,0] = a[j,0]
        x_avg[i,0] = compute_integral_x_T(Ai, a[i,0], x0[i,0], T) / T
        a[j,0] = A[j,i]*x_avg[i,0]
        x_a[i,0] = compute_x_T(Ai, a[i,0], x0[i,0], T)
        
        e0 = T*abs(a[0,0]-a_old[0,0])
        e1 = T*abs(a[1,0]-a_old[1,0])
        print("\nIter %d:"%(count))
        print('approximate x(T)=', x_a.T)
        print('x avg = \t', x_avg.T)
        print("Err est: \t%.3f %.3f"%(e0, e1))
        count += 1
        
        if( e0<thr and e1<thr):
            break
        if count>max_count:
            print("Max iter reached")
            break
    
    for i in range(n):
        Ai = A[i:i+1,i:i+1]
        x_a[i,0] = compute_x_T(Ai, a[i,0], x0[i,0], T)
    print('approximate x=', x_a.T)
    return x
    
#test_random_matrix()
    
A = np.matrix(
[[0.476,  0.015,  0.389,  0.008,  0.015,  0.005,  0.008,  0.028,  0.005,  0.018,  0.028,  0.005, ],
 [0.036,  0.320,  0.003,  0.033,  0.321,  0.006,  0.034,  0.100,  0.012,  0.033,  0.101,  0.001, ],
 [0.132,  0.000,  0.855,  0.001,  0.001,  0.000,  0.001,  0.003,  0.001,  0.002,  0.003,  0.000, ],
 [0.008,  0.014,  0.002,  0.477,  0.013,  0.394,  0.018,  0.028,  0.008,  0.008,  0.029,  0.002, ],
 [0.035,  0.320,  0.008,  0.031,  0.320,  0.005,  0.034,  0.100,  0.012,  0.033,  0.101,  0.001, ],
 [0.002,  0.001,  0.000,  0.135,  0.001,  0.857,  0.001,  0.002,  0.000,  0.000,  0.002,  0.000, ],
 [0.006,  0.011,  0.003,  0.014,  0.011,  0.002,  0.478,  0.020,  0.427,  0.005,  0.019,  0.004, ],
 [0.075,  0.110,  0.026,  0.076,  0.111,  0.013,  0.066,  0.211,  0.019,  0.065,  0.211,  0.018, ],
 [0.002,  0.002,  0.001,  0.003,  0.002,  0.000,  0.178,  0.002,  0.806,  0.001,  0.004,  0.001, ],
 [0.014,  0.011,  0.004,  0.006,  0.011,  0.001,  0.005,  0.019,  0.002,  0.479,  0.019,  0.429, ],
 [0.073,  0.108,  0.025,  0.074,  0.109,  0.012,  0.062,  0.207,  0.031,  0.061,  0.207,  0.029, ],
 [0.002,  0.000,  0.000,  0.001,  0.000,  0.000,  0.002,  0.002,  0.001,  0.181,  0.004,  0.808, ]]
 )
x0 = matlib.rand(A.shape[0]).T

try:
    x_sparse = test_algo_1(A, x0)
except:
    pass

x_sparse = test_algo_3(A, x0)
x = expm(A) * x0

#print("")
#print("x(0)    =", x0.T)
#print("x exact =", x.T)
#print("x approx=", x_sparse.T)
#print("error   =", np.abs(x-x_sparse).T)
