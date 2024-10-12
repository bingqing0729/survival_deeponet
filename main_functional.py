"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
from deepxde.data.simulator import SimulatedFunctional
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf

n_train = 200
n_valid = 200
n_test = 200
show_test = 200
n_partition_y = 250
# Load dataset 
setup = 2
generator = SimulatedFunctional(setup=setup)

N = 1
n_init = 1

x_list, y_list = generator.generate_test(n_test)
F_true, fine_partition = generator.empirical_cdf(x_list,n_check=show_test)

F_all = np.zeros([N,n_test,len(fine_partition)])
l2_F_all = np.zeros([N,show_test,len(fine_partition)])

mse_list = []
medse_list = []
coverage_rate_90_list = []
coverage_rate_95_list = []
l2_coverage_rate_90_list = []
l2_coverage_rate_95_list = []
l2_mse_list = []
l2_medse_list = []


for i in range(N):
    
    print("****** Simulation: ", i+1, "********")
    min_loss = np.inf
    dim_y = 1
    net = dde.nn.DeepONetFunctional(
        [32,32,10],
        [dim_y, 128, 10],
        "relu", 
        "Glorot normal",
        cnn = True
    )

    X, loc, y, ind, t_train, x_list_train, y_list_train = generator.generate_data(n_train,seed=i+1,n_partition_y=n_partition_y)
    dt = np.diff(t_train,prepend=t_train[0])
    X_train = (X.astype(np.float32), loc.astype(np.float32))
    y_train = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))

    X_mean = np.mean(X_train[0],axis=0)
    X_std = np.std(X_train[0],axis=0)
    loc_mean = np.mean(X_train[1],axis=0)
    loc_std = np.std(X_train[1],axis=0)

    X_train = ((X.astype(np.float32)-X_mean)/(X_std+0.00001), (loc.astype(np.float32)-loc_mean)/(loc_std+0.00001))

    X, loc, y, ind, t_valid, x_list_valid, y_list_valid = generator.generate_data(n_valid,seed=N+i+1,n_partition_y=n_partition_y)
    dt = np.diff(t_valid,prepend=t_valid[0])
    #X_valid = (X.astype(np.float32),loc.astype(np.float32))
    X_valid = ((X.astype(np.float32)-X_mean)/(X_std+0.00001), (loc.astype(np.float32)-loc_mean)/(loc_std+0.00001))
    y_valid = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))

    X, loc = generator.expand_test(x_list,t_train)
    dt_test = np.diff(t_train,prepend=t_train[0])
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
        model_temp.compile("adam", lr=0.0001)
        losshistory, train_state = model_temp.train(iterations=20000,batch_size=1000)
        if min_loss > min(losshistory.loss_test):
            model = model_temp
            min_loss = min(losshistory.loss_test)
    
    #model.save("saved_model/model_{}".format(i))

    
    hazard = np.exp(model.predict(X_test))
    hazard = np.reshape(hazard, (n_test,len(dt_test)))
    cum_hazard = np.cumsum(hazard*dt_test,1)
    S =  np.exp(-cum_hazard)
    F = 1-(n_train-1)/(n_train)*S
    F_all[i] = [np.interp(fine_partition, t_train, F[k]) for k in range(n_test)]

    lower_95 = [t_train[k] for k in np.argmax(F > 0.025, axis=1)]
    lower_90 = [t_train[k] for k in np.argmax(F > 0.05, axis=1)]
    upper_95 = [t_train[k] if k!=0 else t_train[-1] for k in np.argmax(F > 0.975, axis=1)]
    upper_90 = [t_train[k] if k!=0 else t_train[-1] for k in np.argmax(F > 0.95, axis=1)]


    coverage_rate_90 = np.mean([(y_list[k] >= lower_90[k]) & (y_list[k] <= upper_90[k]) for k in range(n_test)])
    coverage_rate_90_list.append(coverage_rate_90)
    coverage_rate_95 = np.mean([(y_list[k] >= lower_95[k]) & (y_list[k] <= upper_95[k]) for k in range(n_test)])
    coverage_rate_95_list.append(coverage_rate_95)

    f = np.diff(F,axis=1,prepend=0)
    mean_pred = np.sum(f*t_train,axis=1)
    mse = np.mean((mean_pred-np.array(y_list))**2)
    medse = np.median((mean_pred-np.array(y_list))**2)
    mse_list.append(mse)
    medse_list.append(medse)

    #print("memory: GB", psutil.virtual_memory().used / 2 ** 30)
    
    tf.keras.backend.clear_session()
    del model


    ############# L2 model ###################
    x_train, y_train = np.array(x_list_train), np.array(y_list_train)
    x_valid, y_valid = np.array(x_list_valid), np.array(y_list_valid)
    
    l2_model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=16,kernel_size=3,activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=8,strides=8),
                tf.keras.layers.Conv1D(filters=16,kernel_size=3,activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=8,strides=8),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(1)
                ])
    
    # Compile the model
    l2_model.compile(optimizer='adam',
                loss='mse',
                metrics=['mse'])

    # Train the model
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    l2_model.fit(x_train, y_train, epochs=2000, batch_size=2000, validation_data=(x_valid, y_valid), callbacks=[early_stop])

    l2_pred = l2_model.predict(np.array(x_list))[:,0]
    res = np.array(y_list) -  l2_pred
    res_train = np.array(y_list_train) - l2_model.predict(np.array(x_list_train))[:,0]
    l2_mse = np.mean(res**2)
    l2_medse = np.median(res**2)

    l2_mse_list.append(l2_mse)
    l2_medse_list.append(l2_medse)
    
    l2_F = np.array([[sum((l2_pred[k]+res_train)<fine_partition[j])/n_train for j in range(len(fine_partition))] for k in range(show_test)])
    l2_F_all[i] = l2_F

    lower_95 = [fine_partition[k] for k in np.argmax(l2_F > 0.025, axis=1)]
    lower_90 = [fine_partition[k] for k in np.argmax(l2_F > 0.05, axis=1)]
    upper_95 = [fine_partition[k] if k!=0 else fine_partition[-1] for k in np.argmax(l2_F > 0.975, axis=1)]
    upper_90 = [fine_partition[k] if k!=0 else fine_partition[-1] for k in np.argmax(l2_F > 0.95, axis=1)]


    coverage_rate_90 = np.mean([(y_list[k] >= lower_90[k]) & (y_list[k] <= upper_90[k]) for k in range(n_test)])
    l2_coverage_rate_90_list.append(coverage_rate_90)
    coverage_rate_95 = np.mean([(y_list[k] >= lower_95[k]) & (y_list[k] <= upper_95[k]) for k in range(n_test)])
    l2_coverage_rate_95_list.append(coverage_rate_95)

    tf.keras.backend.clear_session()
    del l2_model


print('mse:', np.mean(mse_list), ' medse:', np.mean(medse_list), ' l2 mse:', np.mean(l2_mse_list), ' l2 medse:', np.mean(l2_medse_list),
      ' cov_90:', np.mean(coverage_rate_90_list), ' cov_95:', np.mean(coverage_rate_95_list), 
      'l2 cov_90:', np.mean(l2_coverage_rate_90_list), 'l2 cov_95:', np.mean(l2_coverage_rate_95_list))

mse_dist = np.mean((F_all-F_true)**2)
l2_mse_dist = np.mean((l2_F_all-F_true)**2)
medse_dist = np.mean(np.median((F_all-F_true)**2,axis=1))
l2_medse_dist = np.mean(np.median((l2_F_all-F_true)**2,axis=1))

print('mse_dist:', mse_dist, 'l2_mse_dist:', l2_mse_dist)
print('medse_dist:', medse_dist, 'l2_medse_dist:', l2_medse_dist)

for i in range(10):
    # original
    plt.plot(fine_partition,np.mean(l2_F_all,axis=0)[i],color='green',linestyle='solid',label='l2')
    plt.plot(fine_partition,np.mean(F_all,axis=0)[i],color='orange',linestyle='solid', label='new')
    plt.plot(fine_partition,F_true[i],color='black',linestyle='solid',label='truth')
    plt.plot(fine_partition,np.percentile(F_all,5,axis=0)[i],color='orange',linestyle='dashed')
    plt.plot(fine_partition,np.percentile(F_all,95,axis=0)[i],color='orange',linestyle='dashed')
    if i == 0: plt.legend()
    plt.savefig("results/figure_{}".format(i))
    plt.close()