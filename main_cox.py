import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from cox import reformat_data, cox_pred

n_train = 2000
n_test = 10

N = 100

loaded_data = np.load('saved_data_{}/test.npz'.format(n_train))
X_test_trt = (loaded_data['X_test_trt'], loaded_data['X_test_trt_loc'])
X_test_ctr = (loaded_data['X_test_ctr'], loaded_data['X_test_ctr_loc'])
S_true_trt, S_true_ctr= loaded_data['S_true_trt'], loaded_data['S_true_ctr']
fine_partition = loaded_data['fine_partition']
t = loaded_data['t']
dt_test = np.diff(t,prepend=0)


# draw on tau/m partition
S_all_trt = np.zeros((N,n_test,len(dt_test)))
S_all_ctr = np.zeros((N,n_test,len(dt_test)))

for i in range(N):
    
    print("****** Simulation: ", i+1, "********")


    loaded_data = np.load('saved_data_{}/train_{}.npz'.format(n_train,i))
    X_train = (loaded_data['X_train_0'], loaded_data['X_train_1'])
    y_train = (loaded_data['y_train_0'], loaded_data['y_train_1'], loaded_data['y_train_2'])
    

    data = reformat_data(X_train,y_train)
    test_data = reformat_data(X_test_trt)

    test_data, beta = cox_pred(data,test_data)
    S_all_trt[i] = np.array(test_data['pred_survival_cox']).reshape([n_test,-1])
    


for i in range(n_test):

    # original
    plt.plot(t,np.mean(S_all_trt,axis=0)[i],color='blue',linestyle='solid',label="cox")
    plt.plot(fine_partition,S_true_trt[i],color='black',linestyle='solid',label="truth")
    if i == 0: plt.legend()
    plt.savefig("results/cox_figure_{}".format(i))
    plt.close()
