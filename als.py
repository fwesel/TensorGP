import numpy as np
import TN_mutable
import approximate_MAP
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge  import KernelRidge
from utils import hilbert_gp_gaussian_prior, find_max_TT_rank, hilbert_GP_RBF, hilbert_GP_uniform, KRR_uniform, hilbert_gp_features, hilbert_gp_features_numpy

file_names = ["airfoil","yacht","energy","concrete"]
folder = "C:\\Users\LocalAdmin\\surfdrive\\Code\\TensorGP\\"
# Loop over filenames
for file_name in file_names:
    file_path = folder+file_name
    X = np.loadtxt(file_path+".csv", delimiter=';')
    y = X[:,-1]
    X = X[:,:-1]
    N, D = X.shape
    print("Processing: ",file_name,"N:",N,"D:",D)


    # Preprocess
    X  = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))

    # Select the number of basis functions
    M = 10
    number_sweeps = 20
    trials = 10

    # # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # # # Kernel Ridge Regression
    param_grid = {'alpha': np.power(10.0,np.arange(-5,5)), 'L': 1+np.power(2.0,np.arange(-2,4))}
    grid_search = GridSearchCV(estimator=KRR_uniform(M=M),param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
    # Fit
    grid_search.fit(X_train, y_train)
    best_alpha = grid_search.best_params_['alpha']
    best_L = grid_search.best_params_['L']
    print("Best alpha:", best_alpha)
    print("Best L:", best_L)
    # Train final model
    final_model = KRR_uniform(M=M,L=best_L,alpha=best_alpha)
    final_model.fit(X_train, y_train)
    # Predict final model
    test_KRR_error = mean_squared_error(y_test,final_model.predict(X_test))
    train_KRR_error = mean_squared_error(y_train,final_model.predict(X_train))
    print("Test KRR error (unconstrained TNKM): ",test_KRR_error)
    
    # Define prior with given hyperparameters
    prior_covariance = np.eye(M)

    # CPD
    R_CPD = np.arange(1,11)

    # # Initialize containers
    P_CPD = M*R_CPD*D
    train_RMSE_CPD_new_regularization = np.zeros((R_CPD.shape[0],trials))
    train_RMSE_CPD_standard_regularization = np.zeros((R_CPD.shape[0],trials))
    test_RMSE_CPD_new_regularization = np.zeros((R_CPD.shape[0],trials))
    test_RMSE_CPD_standard_regularization = np.zeros((R_CPD.shape[0],trials))

    # print("CPD")
    for idx, R in enumerate(R_CPD):
        for trial in range(trials):
            # Standard regularization term
            try:
                cpd_model = approximate_MAP.CPD(X_train,y_train.flatten(),lambda X:hilbert_gp_features_numpy(X,best_L,M),prior_covariance,R,np.sqrt(best_alpha))
                cpd_model.standard_regularization_MAP(random_seed=trial,number_sweeps=number_sweeps)
                test_cpd_predictions = cpd_model.predict(X_test)
                train_cpd_predictions = cpd_model.predict(X_train)
                test_RMSE_CPD_standard_regularization[idx,trial] = mean_squared_error(y_test,test_cpd_predictions)
                train_RMSE_CPD_standard_regularization[idx,trial] = mean_squared_error(y_train,train_cpd_predictions)
            except Exception:
                test_RMSE_CPD_standard_regularization[idx,trial] = np.nan
                train_RMSE_CPD_standard_regularization[idx,trial] = np.nan


            # New regularization
            try:
                cpd_model = approximate_MAP.CPD(X_train,y_train.flatten(),lambda X:hilbert_gp_features_numpy(X,best_L,M),prior_covariance,R,np.sqrt(best_alpha))
                cpd_model.new_regularization_MAP(random_seed=trial,number_sweeps=number_sweeps) 
                test_cpd_predictions = cpd_model.predict(X_test)
                train_cpd_predictions = cpd_model.predict(X_train)
                test_RMSE_CPD_new_regularization[idx,trial] = mean_squared_error(y_test,test_cpd_predictions)
                train_RMSE_CPD_new_regularization[idx,trial] = mean_squared_error(y_train,train_cpd_predictions)
            except Exception:
                test_RMSE_CPD_new_regularization[idx,trial] = np.nan
                train_RMSE_CPD_new_regularization[idx,trial] = np.nan


            print("Trial",str(trial+1),"CPD rank:",str(R),"RMSE new regularization:",str(test_RMSE_CPD_new_regularization[idx,trial]),"RMSE standard regularization:",str(test_RMSE_CPD_standard_regularization[idx,trial]))
        mean_RMSE_CPD_standard_regularization = np.round(np.nanmean(test_RMSE_CPD_standard_regularization[idx,:]),5)
        std_RMSE_CPD_standard_regularization = np.round(np.nanstd(test_RMSE_CPD_standard_regularization[idx,:]),5)
        mean_RMSE_CPD_new_regularization = np.round(np.nanmean(test_RMSE_CPD_new_regularization[idx,:]),5)
        std_RMSE_CPD_new_regularization = np.round(np.nanstd(test_RMSE_CPD_new_regularization[idx,:]),5)
        print("CPD rank:",str(R),"mean RMSE new regularization:",str(mean_RMSE_CPD_new_regularization),"std",str(std_RMSE_CPD_new_regularization), \
              "mean RMSE standard regularization:",str(mean_RMSE_CPD_standard_regularization),"std",str(std_RMSE_CPD_standard_regularization))
        

    # TT
    max_parameters = M*R_CPD[-1]*D
    max_R_TT, n_iter_TT = find_max_TT_rank(max_parameters,D,M)
    P_TT = np.empty(0)
    train_RMSE_TT_new_regularization = np.zeros((n_iter_TT,trials))
    train_RMSE_TT_standard_regularization =  np.zeros((n_iter_TT,trials))
    test_RMSE_TT_new_regularization = np.zeros((n_iter_TT,trials))
    test_RMSE_TT_standard_regularization =  np.zeros((n_iter_TT,trials))

    print("TT")
    print("Max TT-rank:",max_R_TT)
    R = np.ones((D+1,),dtype=np.int64)
    idx_P = 0
    while not np.array_equal(R,max_R_TT):
        for idx_R in range(1,D):
            for trial in range(trials):
                # Standard regularization term
                try:
                    tt_model = approximate_MAP.TT(X_train,y_train.flatten(),lambda X:hilbert_gp_features_numpy(X,best_L,M),prior_covariance,R,np.sqrt(best_alpha))
                    tt_model.standard_regularization_MAP(random_seed=trial,number_sweeps=number_sweeps)
                    test_tt_predictions = tt_model.predict(X_test)
                    train_tt_predictions = tt_model.predict(X_train)
                    test_RMSE_TT_standard_regularization[idx_P,trial] = mean_squared_error(y_test,test_tt_predictions)
                    train_RMSE_TT_standard_regularization[idx_P,trial] = mean_squared_error(y_train,train_tt_predictions)

                except Exception:
                    test_RMSE_TT_standard_regularization[idx_P,trial] = np.nan
                    train_RMSE_TT_standard_regularization[idx_P,trial] = np.nan

                # New regularization term
                try:
                    tt_model = approximate_MAP.TT(X_train,y_train.flatten(),lambda X:hilbert_gp_features_numpy(X,best_L,M),prior_covariance,R,np.sqrt(best_alpha))
                    tt_model.new_regularization_no_site_k_MAP(random_seed=trial,number_sweeps=number_sweeps)
                    test_tt_predictions = tt_model.predict(X_test)
                    train_tt_predictions = tt_model.predict(X_train)
                    test_RMSE_TT_new_regularization[idx_P,trial] = mean_squared_error(y_test,test_tt_predictions)
                    train_RMSE_TT_new_regularization[idx_P,trial] = mean_squared_error(y_train,train_tt_predictions)
                except Exception:
                    test_RMSE_TT_new_regularization[idx_P,trial] = np.nan
                    train_RMSE_TT_new_regularization[idx_P,trial] = np.nan
                print("Trial",str(trial+1),"TT rank:",str(R),"Test RMSE new regularization:",str(test_RMSE_TT_new_regularization[idx_P,trial]),"Test RMSE standard regularization:",str(test_RMSE_TT_standard_regularization[idx_P,trial]))
                print("Trial",str(trial+1),"TT rank:",str(R),"Train RMSE new regularization:",str(train_RMSE_TT_new_regularization[idx_P,trial]),"Train RMSE standard regularization:",str(train_RMSE_TT_standard_regularization[idx_P,trial]))

            mean_RMSE_TT_standard_regularization = np.round(np.nanmean(test_RMSE_TT_standard_regularization[idx_P,:]),5)
            std_RMSE_TT_standard_regularization = np.round(np.nanstd(test_RMSE_TT_standard_regularization[idx_P,:]),5)
            mean_RMSE_TT_new_regularization = np.round(np.nanmean(test_RMSE_TT_new_regularization[idx_P,:]),5)
            std_RMSE_TT_new_regularization = np.round(np.nanstd(test_RMSE_TT_new_regularization[idx_P,:]),5)
            print("TT rank:",str(R),"mean RMSE new regularization:",str(mean_RMSE_TT_new_regularization),"std",str(std_RMSE_TT_new_regularization), \
              "mean RMSE standard regularization:",str(mean_RMSE_TT_standard_regularization),"std",str(std_RMSE_TT_standard_regularization))
            
            P_TT = np.append(P_TT,tt_model.P)
            R[idx_R] += 1
            idx_P += 1
    # Save results
    np.savez(file_name+".npz",train_KRR_error=train_KRR_error,test_KRR_error=test_KRR_error,\
             test_RMSE_CPD_standard_regularization=test_RMSE_CPD_standard_regularization,train_RMSE_CPD_standard_regularization=train_RMSE_CPD_standard_regularization,\
             test_RMSE_CPD_new_regularization=test_RMSE_CPD_new_regularization,train_RMSE_CPD_new_regularization=train_RMSE_CPD_new_regularization,\
             test_RMSE_TT_standard_regularization=test_RMSE_TT_standard_regularization,train_RMSE_TT_standard_regularization=train_RMSE_TT_standard_regularization,\
             test_RMSE_TT_new_regularization=test_RMSE_TT_new_regularization,train_RMSE_TT_new_regularization=train_RMSE_TT_new_regularization,\
            P_CPD=P_CPD,P_TT=P_TT,N=N,D=D,N_train=X_train.shape[0])