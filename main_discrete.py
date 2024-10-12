"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
from deepxde.data.simulator import SimulatedDiscrete
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf

n_train = 2000
n_valid = 2000
n_test = 10
# Load dataset 
generator = SimulatedDiscrete()

N = 100
n_init = 3

x_list, y_list = generator.generate_test(n_test)
F_true, fine_partition = generator.empirical_cdf(x_list)

F_all = np.zeros([N,n_test,len(fine_partition)])


for i in range(N):
    
    print("****** Simulation: ", i+1, "********")
    min_loss = np.inf
    dim_y = 1
    net = dde.nn.DeepONetFunctional(
        [128,128,10],
        [dim_y, 128, 10],
        "relu", 
        "Glorot normal",
        cnn = True
    )

    X, loc, y, ind, t_train = generator.generate_data(n_train,seed=i+1)
    dt = np.diff(t_train,prepend=t_train[0])
    X_train = (X.astype(np.float32), loc.astype(np.float32))
    y_train = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))

    X_mean = np.mean(X_train[0],axis=0)
    X_std = np.std(X_train[0],axis=0)
    loc_mean = np.mean(X_train[1],axis=0)
    loc_std = np.std(X_train[1],axis=0)

    X_train = ((X.astype(np.float32)-X_mean)/(X_std+0.00001), (loc.astype(np.float32)-loc_mean)/(loc_std+0.00001))

    X, loc, y, ind, t_valid = generator.generate_data(n_valid,seed=N+i+1)
    dt = np.diff(t_valid,prepend=t_valid[0])
    #X_valid = (X.astype(np.float32),loc.astype(np.float32))
    X_valid = ((X.astype(np.float32)-X_mean)/(X_std+0.00001), (loc.astype(np.float32)-loc_mean)/(loc_std+0.00001))
    y_valid = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))

    X, loc = generator.expand_test(x_list,fine_partition)
    #X_test = (X.astype(np.float32),loc.astype(np.float32))
    X_test = ((X.astype(np.float32)-X_mean)/(X_std+0.00001), (loc.astype(np.float32)-loc_mean)/(loc_std+0.00001))

    data = dde.data.Triple(
        X_train=X_train, y_train=y_train, X_test=X_valid, y_test=y_valid
    )

    min_loss = np.inf
    for _ in range(n_init):

        # Define a Model
        model_temp = dde.Model(data, net)

        # Compile and Train
        model_temp.compile("adam", lr=0.0001,loss="custom_discrete")
        losshistory, train_state = model_temp.train(iterations=20000,batch_size=1000)
        if min_loss > min(losshistory.loss_test):
            model = model_temp
            min_loss = min(losshistory.loss_test)
    
    #model.save("saved_model/model_{}".format(i))

    
    h = np.exp(model.predict(X_test))
    h = np.reshape(h, (n_test,len(fine_partition)))
    prod_h = np.cumprod(1+h,1)
    frac = h/prod_h
    F = np.cumsum(frac,1)
    F_all[i] = [F[k] for k in range(n_test)]
    
    
    tf.keras.backend.clear_session()
    del model

for i in range(n_test):
    # original
    plt.step(fine_partition,np.mean(F_all,axis=0)[i],color='orange',linestyle='solid')
    plt.step(fine_partition,F_true[i],color='black',linestyle='solid')
    plt.step(fine_partition,np.percentile(F_all,5,axis=0)[i],color='orange',linestyle='dashed')
    plt.step(fine_partition,np.percentile(F_all,95,axis=0)[i],color='orange',linestyle='dashed')
    plt.savefig("results/figure_{}".format(i))
    plt.close()