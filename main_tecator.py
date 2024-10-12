from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import deepxde as dde
import numpy as np
from sklearn.model_selection import KFold
from deepxde.data.simulator import expand_data
from sklearn.metrics import r2_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the ARFF file
data, meta = arff.loadarff('C:\\Users\\bingq\\OneDrive\\Desktop\\tecator.arff')

# Convert the data to a pandas DataFrame
data = pd.DataFrame(data)

# Extract all columns starting with "absorbance" into a list called 'absorb'
absorb_columns = [col for col in data.columns if col.startswith('absorbance')]
data['absorb'] = data[absorb_columns].values.tolist()

# Select only the 'absorb', 'moisture', and 'fat' columns
data = data[['absorb', 'moisture', 'fat']]

# Standardize the 'moisture' column
scaler = StandardScaler()
data['moisture'] = scaler.fit_transform(data[['moisture']])
data['fat'] = scaler.fit_transform(data[['fat']])

kf = KFold(n_splits=10,shuffle=True, random_state=42) 

x_absorb = data['absorb'].to_numpy()
moisture = data['moisture'].to_numpy()
fat = data['fat'].to_numpy()

mean_pred_all = np.array([])
l2_mean_pred_all = np.array([])
y_test_all = np.array([])
lower_95 = np.array([])
lower_90 = np.array([])
upper_95 = np.array([])
upper_90 = np.array([])

for train_index, test_index in kf.split(data):

    train_index, valid_index = train_index[:len(train_index)//2], train_index[len(train_index)//2:]
    n_train = len(train_index)
    n_test = len(test_index)

    temp_train, temp_valid, temp_test = x_absorb[train_index], x_absorb[valid_index], x_absorb[test_index]
    w_train, w_valid, w_test = moisture[train_index], moisture[valid_index], moisture[test_index]
    y_train, y_valid, y_test = fat[train_index], fat[valid_index], fat[test_index]

    ############# L2 model ###################
    
    x_train = np.stack((np.stack(temp_train), np.repeat(w_train[:, np.newaxis], 100, axis=1)), axis=-1)
    x_valid = np.stack((np.stack(temp_valid), np.repeat(w_valid[:, np.newaxis], 100, axis=1)), axis=-1)
    x_test = np.stack((np.stack(temp_test), np.repeat(w_test[:, np.newaxis], 100, axis=1)), axis=-1)

    l2_model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=32,kernel_size=3,activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2,strides=2),
                tf.keras.layers.Conv1D(filters=32,kernel_size=3,activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2,strides=2),
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
    l2_model.fit(x_train, y_train, epochs=2000, batch_size=1000, validation_data=(x_valid, y_valid), callbacks=[early_stop])

    l2_mean_pred = l2_model.predict(np.array(x_test))[:,0]
    l2_mean_pred_all = np.append(l2_mean_pred_all, l2_mean_pred)
    y_test_all = np.append(y_test_all,y_test)

    ################### new model #############################
    X, loc, y, ind, t_train = expand_data(temp_train,w_train,y_train,n_partition_y=500)
    dt = np.diff(t_train,prepend=t_train[0])
    X_train = (X.astype(np.float32), loc.astype(np.float32))
    y_train = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))
    
    X, loc, y, ind, t_valid = expand_data(temp_valid,w_valid,y_valid,n_partition_y=500)
    dt = np.diff(t_valid,prepend=t_valid[0])
    X_valid = (X.astype(np.float32), loc.astype(np.float32))
    y_valid = (y.astype(np.float32), ind.astype(np.float32), dt.astype(np.float32))

    net = dde.nn.DeepONetFunctional(
        [32,32,10],
        [1, 128, 10],
        "relu", 
        "Glorot normal",
        cnn = True
    )

    data = dde.data.Triple(
            X_train=X_train, y_train=y_train, X_test=X_valid, y_test=y_valid
        )
    model = dde.Model(data, net)

    # Compile and Train
    model.compile("adam", lr=0.0001)
    losshistory, train_state = model.train(iterations=20000,batch_size=1000)

    X, loc, y, ind, t_test = expand_data(temp_test,w_test,y_test,partition_y=t_train)
    dt = np.diff(t_test,prepend=t_test[0])
    X_test = (X.astype(np.float32), loc.astype(np.float32))

    hazard = np.exp(model.predict(X_test))
    hazard = np.reshape(hazard, (n_test,len(dt)))
    cum_hazard = np.cumsum(hazard*dt,1)
    S =  np.exp(-cum_hazard)
    F = 1-(n_train-1)/(n_train)*S

    lower_95 = np.append(lower_95,[t_train[k] for k in np.argmax(F > 0.025, axis=1)])
    lower_90 = np.append(lower_90,[t_train[k] for k in np.argmax(F > 0.05, axis=1)])
    upper_95 = np.append(upper_95,[t_train[k] for k in np.argmax(F > 0.975, axis=1)])
    upper_90 = np.append(upper_90,[t_train[k] for k in np.argmax(F > 0.95, axis=1)])

    pmf = np.diff(F,axis=1,prepend=0)
    mean_pred = np.sum(pmf*t_test,axis=1)

    mean_pred_all = np.append(mean_pred_all, mean_pred)
    #for j in range(len(F)):
        #plt.plot(t_test,F[j])
        #plt.axvline(x=y_test[j], color='r', linestyle='--', linewidth=2)
        #plt.show()

    
mse = np.mean((mean_pred_all-y_test_all)**2)
medse = np.median((mean_pred_all-y_test_all)**2)
l2_mse = np.mean((l2_mean_pred_all-y_test_all)**2)
l2_medse = np.median((l2_mean_pred_all-y_test_all)**2)
print("mse:", mse, " medse:", medse, " l2_mse:", l2_mse, " l2_medse:", l2_medse)

cov_90 = np.mean((y_test_all >= lower_90) & (y_test_all <= upper_90))
cov_95 = np.mean((y_test_all >= lower_95) & (y_test_all <= upper_95))

print("cov_90:", cov_90, " cov_95:", cov_95)

r2 = r2_score(y_test_all, mean_pred_all)
l2_r2 = r2_score(y_test_all,l2_mean_pred_all)
print("R^2:", r2)
print("L2 R^2:", l2_r2)

plt.plot(y_test_all,mean_pred_all,'.')
plt.plot(y_test_all,y_test_all)
plt.show()

plt.plot(y_test_all,l2_mean_pred_all,'.')
plt.plot(y_test_all,y_test_all)
plt.show()