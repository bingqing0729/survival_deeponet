"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
from deepxde.data.simulator import SimulatedSurvival
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

n_train = 500
n_valid = 500
n_test = 10

m = 250
tau = 100
fine_grid_step = 0.1
num_node = 256
lr = 0.001
batch_size = 1000

generator = SimulatedSurvival(tau,fine_grid_step)

N = 1
n_init = 1

"""
# t is tau/m partition, fine_partition is fixed partition
X, loc, t, S_true_trt, fine_partition = generator.generate_data(n_test,m=m,seed=0,test=True,trt=1)
dt_test = np.diff(t,prepend=0)
X_test_trt = (X.astype(np.float32), loc.astype(np.float32))

X, loc, t, S_true_ctr, fine_partition = generator.generate_data(n_test,m=m,seed=0,test=True,trt=0)
dt_test = np.diff(t,prepend=0)
X_test_ctr = (X.astype(np.float32), loc.astype(np.float32))

np.savez('saved_data_{}_500/test.npz'.format(n_train), X_test_trt=X_test_trt[0], X_test_trt_loc=X_test_trt[1],
         S_true_trt=S_true_trt, X_test_ctr=X_test_ctr[0], X_test_ctr_loc=X_test_ctr[1],
         S_true_ctr=S_true_ctr,fine_partition=fine_partition,t=t)
"""
loaded_data = np.load('saved_data_{}_500/test.npz'.format(n_train))
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

    dim_y = 1
    net = dde.nn.DeepONet(
        [m+2, num_node, 10],
        [dim_y, num_node, 10],
        "relu", 
        "Glorot normal",
        cnn = True
    )
    """
    X, loc, y, ind, t_train = generator.generate_data(n_train,m=m,seed=i+1+2*N)
    dt = np.diff(t_train,prepend=0)
    X_train = (X.astype(np.float32), loc.astype(np.float32))
    y_train = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))

    X, loc, y, ind, t_valid = generator.generate_data(n_valid,m=m,seed=3*N+i+1)
    dt = np.diff(t_valid,prepend=0)
    X_valid = (X.astype(np.float32), loc.astype(np.float32))
    y_valid = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))

    np.savez('saved_data_{}_500/train_{}.npz'.format(n_train,i), X_train_0=X_train[0], X_train_1=X_train[1],
         y_train_0=y_train[0], y_train_1=y_train[1], y_train_2=y_train[2])
    np.savez('saved_data_{}_500/valid_{}.npz'.format(n_train,i), X_valid_0=X_valid[0], X_valid_1=X_valid[1],
         y_valid_0=y_valid[0], y_valid_1=y_valid[1], y_valid_2=y_valid[2])
    """
    loaded_data = np.load('saved_data_{}_500/train_{}.npz'.format(n_train,i))
    X_train = (loaded_data['X_train_0'], loaded_data['X_train_1'])
    y_train = (loaded_data['y_train_0'], loaded_data['y_train_1'], loaded_data['y_train_2'])
    loaded_data = np.load('saved_data_{}_500/valid_{}.npz'.format(n_train,i))
    X_valid = (loaded_data['X_valid_0'], loaded_data['X_valid_1'])
    y_valid = (loaded_data['y_valid_0'], loaded_data['y_valid_1'], loaded_data['y_valid_2'])
    

    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaler1.fit(X_train[0])
    scaler2.fit(X_train[1])

    X_train = (scaler1.transform(X_train[0]),scaler2.transform(X_train[1]))
    X_valid = (scaler1.transform(X_valid[0]),scaler2.transform(X_valid[1]))

    X_test_trt_now = (scaler1.transform(X_test_trt[0]),scaler2.transform(X_test_trt[1]))
    X_test_ctr_now = (scaler1.transform(X_test_ctr[0]),scaler2.transform(X_test_ctr[1]))


    data = dde.data.Triple(
        X_train=X_train, y_train=y_train, X_test=X_valid, y_test=y_valid
    )

    min_loss = np.inf
    for k in range(n_init):
        # Define a Model
        model_temp = dde.Model(data, net)

        # Compile and Train
        model_temp.compile("adam", lr=lr)
        losshistory, train_state = model_temp.train(iterations=20000,batch_size=batch_size)

        if min_loss > min(losshistory.loss_test):
            model = model_temp
            min_loss = min(losshistory.loss_test)
    
    #model.save("saved_model/model_{}.weights.h5".format(i))

    # Plot the loss trajectory
    #dde.utils.plot_loss_history(losshistory)
    #plt.show()

    hazard = np.exp(model.predict(X_test_trt_now))
    hazard = np.reshape(hazard, (n_test,len(dt_test)))
    cum_hazard = np.cumsum(hazard*dt_test,1)
    S =  np.exp(-cum_hazard)
    S_all_trt[i] = S

    hazard = np.exp(model.predict(X_test_ctr_now))
    hazard = np.reshape(hazard, (n_test,len(dt_test)))
    cum_hazard = np.cumsum(hazard*dt_test,1)
    S =  np.exp(-cum_hazard)
    S_all_ctr[i] = S

    with open("results/S_trt.pkl",'wb') as f:
        pickle.dump(S_all_trt,f)
    with open("results/S_ctr.pkl",'wb') as f:
        pickle.dump(S_all_ctr,f)

    print("memory: GB", psutil.virtual_memory().used / 2 ** 30)
    
    tf.keras.backend.clear_session()
    del model

for i in range(10):
    # trt & ctr
    plt.plot(t,np.mean(S_all_trt,axis=0)[i],color='orange',linestyle='solid',label='estimate of control')
    plt.plot(fine_partition,S_true_trt[i],color='black',linestyle='solid',label='truth')
    plt.plot(t,np.percentile(S_all_trt,5,axis=0)[i],color='orange',linestyle='dashed')
    plt.plot(t,np.percentile(S_all_trt,95,axis=0)[i],color='orange',linestyle='dashed')
    plt.plot(t,np.mean(S_all_ctr,axis=0)[i],color='green',linestyle='solid',label='estimate of treatment')
    plt.plot(fine_partition,S_true_ctr[i],color='black',linestyle='solid')
    plt.plot(t,np.percentile(S_all_ctr,5,axis=0)[i],color='green',linestyle='dashed')
    plt.plot(t,np.percentile(S_all_ctr,95,axis=0)[i],color='green',linestyle='dashed')
    if i == 0: plt.legend()
    plt.savefig("results/figure_diff_{}".format(i))
    plt.close()

    # original
    plt.plot(t,np.mean(S_all_trt,axis=0)[i],color='orange',linestyle='solid',label='estimate')
    plt.plot(fine_partition,S_true_trt[i],color='black',linestyle='solid',label='truth')
    plt.plot(t,np.percentile(S_all_trt,5,axis=0)[i],color='orange',linestyle='dashed')
    plt.plot(t,np.percentile(S_all_trt,95,axis=0)[i],color='orange',linestyle='dashed')
    #plt.xlim((-3,40))
    if i == 0: plt.legend()
    plt.savefig("results/figure_{}".format(i))
    plt.close()

