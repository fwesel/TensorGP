import TN_mutable
from utils import kernel, find_max_TT_rank, hilbert_gp_features, hilbert_gp_features_numpy, hilbert_gp_gaussian_prior
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.constants import golden

# Plotting

n_fig_per_row = 4
textwidth_pt = 487.8225
points_per_inch = 72.27
textwidth_inches = textwidth_pt / points_per_inch
image_size = textwidth_inches / n_fig_per_row

N = 1    # this is NOT the number of data points
D = 16
M = 10
lengthscale = 1
# prior_cov = np.identity(M)
prior_cov = hilbert_gp_gaussian_prior(M,lengthscale).eval()
sqrt_prior_covariance = np.sqrt(prior_cov)
feature = lambda x: hilbert_gp_features(x,1,M)
feature_numpy = lambda x: hilbert_gp_features_numpy(x,1,M)


n_restarts = 10
n_samples = 10000
n_bins = int(np.ceil(np.sqrt(n_samples)))

R_CPD = np.array([1,2,5,10,20,50,100,1000,10000])
parameters = R_CPD*M*D

# Initialize containers
ks_CPD = np.zeros((R_CPD.size,n_restarts))
P_CPD = np.zeros((R_CPD.size,))

ks_TT = np.zeros((R_CPD.size,n_restarts))
P_TT =np.zeros((R_CPD.size,))



for n in range(n_restarts):
    print("restart "+str(n))
    np.random.seed(n)
    X = np.random.normal(0,1, size=(N,D))
    y = np.random.normal(0,1, size=(N,))   # placeholder, needed by PyMC
    # Compute sqrt of kernel at random point X
    k = kernel(X,X,feature_numpy,prior_cov).flatten()
    k = k[0]
    normal_CDF = lambda x: stats.norm.cdf(x,loc=0,scale=np.sqrt(k))
    normal_PDF = lambda x: stats.norm.pdf(x,loc=0,scale=np.sqrt(k))
    print("CPD")
    idx_CPD = 0
    for R in R_CPD:
        cpd_model = TN_mutable.CPD(X,y,feature,sqrt_prior_covariance,R,1)
        cpd_samples = cpd_model.sample_prior(n_samples=n_samples,random_seed=0).values.flatten()
        P = cpd_model.P
        ks_test = stats.cramervonmises(cpd_samples, normal_CDF)
        ks_CPD[idx_CPD,n] = ks_test.statistic
        idx_CPD += 1

    print("TT")
    idx_TT = 0
    for P in parameters:
        R, n_iter_TT = find_max_TT_rank(P,D,M)
        tt_model = TN_mutable.TT(X,y,feature,sqrt_prior_covariance,R,1)
        P_TT[idx_TT] = tt_model.P
        tt_samples = tt_model.sample_prior(n_samples=n_samples,random_seed=n).values.flatten()
        ks_test = stats.cramervonmises(tt_samples, normal_CDF)
        ks_TT[idx_TT,n] = ks_test.statistic
        idx_TT += 1
    np.savez("final_convergence_D_"+str(D)+".npz",ks_CPD=ks_CPD,ks_TT=ks_TT,P_CPD=P_CPD,P_TT=P_TT)

    
# plt.figure(figsize=(image_size,image_size))
# plt.errorbar(P_CPD,np.mean(ks_CPD,axis=1),yerr=np.std(ks_CPD,axis=1),fmt='o', label='CPD')
# plt.errorbar(P_TT,np.mean(ks_TT,axis=1),yerr=np.std(ks_TT,axis=1),fmt='^', label='TT')
# plt.legend()
# plt.savefig("convergence_D_"+str(D)+".pdf")