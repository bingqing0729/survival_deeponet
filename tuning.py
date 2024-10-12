"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
from deepxde.data.simulator import SimulatedSurvival
import matplotlib.pyplot as plt
import numpy as np
import itertools


n_train = 1000
n_valid = 1000

m_list = [50,100,150,200,250] #[100,200,500]
num_nodes_list = [64] #[32,64,128]
lr_list = [0.0001]
batch_size_list = [500] #[100,500,1000]
para_list = list(itertools.product(m_list,num_nodes_list,lr_list,batch_size_list))

generator = SimulatedSurvival(100,0.1)

N = 10

for i in range(N):

    with open('results/hyperparameter_all.csv','a') as file:
        file.write('Simulation: '+str(i+1)+'\n')

    print("****** Simulation: ", i+1, "********")
    min_loss = 100
    for (m, num_nodes, lr, batch_size) in para_list:
        # Choose a network
        dim_y = 1
        net = dde.nn.DeepONet(
            [m+2, num_nodes, num_nodes, 10],
            [dim_y, num_nodes, 10],
            "relu", 
            "Glorot normal",
            cnn = False
        )

        X, loc, y, ind, t_train = generator.generate_data(n_train,m,seed=i)
        loc = loc/100
        dt = np.diff(t_train,prepend=0)
        X_train = (X.astype(np.float32), loc.astype(np.float32))
        y_train = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))

        X, loc, y, ind, t_valid = generator.generate_data(n_train,m,seed=N+i)
        loc = loc/100
        dt = np.diff(t_valid,prepend=0)
        X_valid = (X.astype(np.float32), loc.astype(np.float32))
        y_valid = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))

        X, loc, y, ind, t_test = generator.generate_data(n_train,m,seed=2*N+i)
        loc = loc/100
        dt = np.diff(t_test,prepend=0)
        X_test= (X.astype(np.float32), loc.astype(np.float32))
        y_test = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))


        data = dde.data.Triple(
            X_train=X_train, y_train=y_train, X_test=X_valid, y_test=y_valid
        )


        # Define a Model
        model = dde.Model(data, net)

        # Compile and Train
        model.compile("adam", lr=lr)
        losshistory, train_state = model.train(iterations=20000,batch_size=batch_size)

        log_hazard = model.predict(X_test)
        loss = m * np.mean(np.multiply(np.multiply(np.exp(log_hazard),y_test[2]),y_test[1])-np.multiply(np.multiply(log_hazard,y_test[0]),y_test[1]))

        with open('results/hyperparameter_all.csv','a') as file:
            #file.write(','.join(map(str,[m, num_nodes, lr, batch_size]))+'\n')
            file.write('batch_size = '+ str(batch_size)+'\n')
            file.write(str(loss)+'\n')
        if min_loss > loss:
            min_loss = loss
            best_para = (m, num_nodes, lr, batch_size)
    

    print("min_loss and best m", min_loss, best_para)
    with open('results/hyperparameter.csv','a') as file:
        file.write(','.join(map(str,best_para))+'\n')
