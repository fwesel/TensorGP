import numpy as np
import matplotlib.pyplot as plt


golden = (1 + 5 ** 0.5) / 2

n_fig_per_row = 2
textwidth_inches = 5.5
image_width = textwidth_inches / n_fig_per_row
image_height = image_width/golden

points_per_inch = 72.27
fontsize = 10


# # plt.rcParams.update({
# #     "text.usetex": True,
# #     "font.family": "serif",  # Choose the same font family as in your LaTeX document
# # })



from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'




from matplotlib.ticker import ScalarFormatter, LogFormatterExponent, StrMethodFormatter, NullFormatter
idx = 0
file_names = ["yacht","airfoil","energy","concrete"]
n_files = len(file_names)
for name in file_names:
    data = np.load(name+".npz")
    P_CPD = data['P_CPD']
    test_RMSE_CPD_standard_regularization = data['test_RMSE_CPD_standard_regularization']
    train_RMSE_CPD_standard_regularization = data['train_RMSE_CPD_standard_regularization']
    test_RMSE_CPD_new_regularization = data['test_RMSE_CPD_new_regularization']
    train_RMSE_CPD_new_regularization = data['train_RMSE_CPD_new_regularization']

    P_TT = data['P_TT']
    test_RMSE_TT_standard_regularization = data['test_RMSE_TT_standard_regularization']
    train_RMSE_TT_standard_regularization = data['train_RMSE_TT_standard_regularization']
    test_RMSE_TT_new_regularization = data['test_RMSE_TT_new_regularization']
    train_RMSE_TT_new_regularization = data['train_RMSE_TT_new_regularization']

    threshold = 10
    temp = P_CPD[threshold]
    temp_TT = P_TT[P_TT<=temp].astype(int)
    threshold_tt = temp_TT.shape[0]
    P_CPD = P_CPD[0:threshold]
    test_RMSE_CPD_standard_regularization = test_RMSE_CPD_standard_regularization[0:threshold,:]
    train_RMSE_CPD_standard_regularization = train_RMSE_CPD_standard_regularization[0:threshold,:]
    test_RMSE_CPD_new_regularization = test_RMSE_CPD_new_regularization[0:threshold,:]
    train_RMSE_CPD_new_regularization = train_RMSE_CPD_new_regularization[0:threshold,:]

    P_TT = P_TT[0:threshold_tt]
    test_RMSE_TT_standard_regularization = test_RMSE_TT_standard_regularization[0:threshold_tt,:]
    train_RMSE_TT_standard_regularization = train_RMSE_TT_standard_regularization[0:threshold_tt,:]
    test_RMSE_TT_new_regularization = test_RMSE_TT_new_regularization[0:threshold_tt,:]
    train_RMSE_TT_new_regularization = train_RMSE_TT_new_regularization[0:threshold_tt,:]
    
    test_KRR_error = data['test_KRR_error']
    train_KRR_error = data['train_KRR_error']

    N = data['N']
    N_train = data['N_train']
    D = data['D']

    # # Train plot
    plt.figure(figsize=(image_width,image_height))

    plt.rc('font', size=fontsize)
    plt.title("train RMSE "+name+", N: "+str(N)+", D: "+str(D))
    plt.grid(True, linestyle='dashed', axis='y', linewidth=0.1, which = 'both', dashes=(10, 5))
    plt.errorbar(P_CPD, np.nanmean(train_RMSE_CPD_new_regularization,axis=1), yerr=np.nanstd(train_RMSE_CPD_new_regularization,axis=1)  ,label="P-CPD",linewidth=0.5)  
    plt.errorbar(P_TT, np.nanmean(train_RMSE_TT_new_regularization,axis=1), yerr=np.nanstd(train_RMSE_TT_new_regularization,axis=1), label="P-TT",linewidth=0.5)  
    plt.errorbar(P_CPD, np.nanmean(train_RMSE_CPD_standard_regularization,axis=1), yerr=np.nanstd(train_RMSE_CPD_standard_regularization,axis=1), label="CPD",linewidth=0.5)  
    plt.errorbar(P_TT, np.nanmean(train_RMSE_TT_standard_regularization,axis=1), yerr=np.nanstd(train_RMSE_TT_standard_regularization,axis=1), label="TT",linewidth=0.5)  
    plt.axhline(y=train_KRR_error, color='black', linestyle='--',label="GP")
    plt.yscale('log')
    if idx in np.arange(n_files-n_fig_per_row,n_files):
        plt.xlabel('$P$')
    if np.mod(idx,n_fig_per_row) == 0:
        plt.ylabel('RMSE')
    # if idx == 0:
        # plt.legend()
        # legend = plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center right', borderaxespad=0.)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True, shadow=True)
    plt.ylim(top=1.1*np.maximum.reduce([np.nanmean(train_RMSE_CPD_new_regularization,axis=1)[0], np.nanmean(train_RMSE_TT_new_regularization,axis=1)[0], np.nanmean(train_RMSE_CPD_standard_regularization,axis=1)[0], np.nanmean(train_RMSE_TT_standard_regularization,axis=1)[0]]))
    plt.savefig("train_"+name+".pdf",bbox_inches='tight',pad_inches = 0)
    plt.close()




    plt.figure(figsize=(image_width,image_height))
    plt.rc('font', size=fontsize)
    plt.title("test RMSE "+name+", N: "+str(N)+", D: "+str(D))
    plt.grid(True, linestyle='dashed', axis='y', linewidth=0.1, which = 'both', dashes=(10, 5))
    plt.errorbar(P_CPD, np.nanmean(test_RMSE_CPD_new_regularization,axis=1), yerr=np.nanstd(test_RMSE_CPD_new_regularization,axis=1)  ,label="P-CPD",linewidth=0.5)  
    plt.errorbar(P_TT, np.nanmean(test_RMSE_TT_new_regularization,axis=1), yerr=np.nanstd(test_RMSE_TT_new_regularization,axis=1), label="P-TT",linewidth=0.5)  
    plt.errorbar(P_CPD, np.nanmean(test_RMSE_CPD_standard_regularization,axis=1), yerr=np.nanstd(test_RMSE_CPD_standard_regularization,axis=1), label="CPD",linewidth=0.5)  
    plt.errorbar(P_TT, np.nanmean(test_RMSE_TT_standard_regularization,axis=1), yerr=np.nanstd(test_RMSE_TT_standard_regularization,axis=1), label="TT",linewidth=0.5)  
    plt.axhline(y=test_KRR_error, color='black', linestyle='--',label="GP")
    plt.yscale('log')
    if np.mod(idx,n_fig_per_row) == 0:
        plt.ylabel('RMSE')
    # if idx == 0:
        # plt.legend()
        # legend = plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center right', ncol=1)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    if idx in np.arange(n_files-n_fig_per_row,n_files):
        plt.xlabel('$P$')
    plt.ylim(top=1.1*np.maximum.reduce([np.nanmean(test_RMSE_CPD_new_regularization,axis=1)[0], np.nanmean(test_RMSE_TT_new_regularization,axis=1)[0], np.nanmean(test_RMSE_CPD_standard_regularization,axis=1)[0], np.nanmean(test_RMSE_TT_standard_regularization,axis=1)[0]]))
    plt.savefig("test_"+name+".pdf",bbox_inches='tight',pad_inches = 0)
    plt.close()

    idx += 1