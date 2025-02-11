import numpy as np
import pymc as pm
import pytensor.tensor as pt
from sklearn.kernel_ridge  import KernelRidge

def khatri_rao(L,R):
    r1,c1 = L.shape   
    r2,c2 = R.shape
    #y=repmat(L,1,c2).*kron(R, ones(1, c1));  
    return pt.tile(L,(1,c2))*pt.slinalg.kron(R, pt.ones((1,c1)))

def khatri_rao_numpy(L,R):
    r1,c1 = L.shape   
    r2,c2 = R.shape
    #y=repmat(L,1,c2).*kron(R, ones(1, c1));  
    return np.tile(L,(1,c2))*np.kron(R, np.ones((1,c1)))

def hilbert_gp_features(x,L,M):
    idx = pt.arange(1,M+1,1)
    matrix  = pt.tile(idx, (x.shape[0],1))
    a = pt.pi*(L+x[:, pt.newaxis])*matrix
    return pt.sin(a/2/L)/pt.sqrt(L)

def hilbert_gp_features_numpy(x,L,M):
    idx = np.arange(1,M+1,1)
    matrix  = np.tile(idx, (x.shape[0],1))
    a = np.pi*(L+x[:, np.newaxis])*matrix
    return np.sin(a/2/L)/np.sqrt(L)

def hilbert_gp_gaussian_prior(M,lengthscale):
    w = pt.arange(1,M+1,1)
    return pt.diag( pt.sqrt(2*pt.pi)*lengthscale*pt.exp(-pt.power(pt.pi*w/2,2)*pt.power(lengthscale,2)/2))

def kernel(x,z,feature,prior_cov):
    x = np.atleast_2d(x)
    z = np.atleast_2d(z)
    D = np.shape(x)[1]
    M = np.shape(prior_cov)[0]
    k = 1.0
    for d in range(D):
        a = feature(x[:,d])
        b = feature(z[:,d])
        k *= np.inner(a, np.matmul( prior_cov,  b.T).T)
    return k

def hilbert_gp_gaussian_kernel(X,Y,lenghtscale,L,M):
    return kernel(X,Y,lambda X: hilbert_gp_features(X,L,M).eval(), hilbert_gp_gaussian_prior(M,lenghtscale))

def hilbert_gp_uniform_kernel(X,Y,L,M):
    return kernel(X,Y,lambda X: hilbert_gp_features_numpy(X,L,M), np.eye(M))


class KRR_uniform(KernelRidge):
    def __init__(self,L=1,M=10,alpha=1):
        self.L = L
        self.M = M
        self.alpha = alpha
        super().__init__(kernel=self.my_kernel,alpha=self.alpha)
    def my_kernel(self, X, Y):
        return hilbert_gp_uniform_kernel(X,Y,self.L,self.M)



def find_max_TT_rank(target_P,D,M):
    new_array = np.ones((D+1,),dtype=np.int64)
    P = M* np.sum(new_array[:-1] * new_array[1:])
    number_iterations = 0
    while P < target_P:
        for idx in range(1,D,1):
            new_array[idx] += 1
            number_iterations += 1
            P = M* np.sum(new_array[:-1] * new_array[1:])
    return new_array, number_iterations


def solve_sqrt_linear_system(C,sqrt_regularization,y):
    lhs = np.vstack([C,sqrt_regularization])
    rhs = np.hstack([y,np.zeros((sqrt_regularization.shape[0],))])
    return np.linalg.lstsq(lhs,rhs[:,np.newaxis],rcond=None)[0]