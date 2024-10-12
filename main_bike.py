import pandas as pd
import matplotlib.pyplot as plt
import deepxde as dde
import numpy as np
from sklearn.model_selection import KFold
from deepxde.data.simulator import expand_data
from sklearn.metrics import r2_score
import tensorflow as tf

n_partition_y = 200

data = pd.read_csv('C:\\Users\\bingq\\OneDrive\\Desktop\\bike_sharing\\bike_sharing_hourly_imputed.csv')

# Convert the 'dteday' column to datetime
data['dteday'] = pd.to_datetime(data['dteday'])

# Group by date
grouped_data = data.groupby(data['dteday'].dt.date)

# Initialize lists to store the extracted information
dates = []
x_temps = []
w_workingday = []
y_cnt = []

# Iterate over each group
for date, group in grouped_data:
    dates.append(date)
    x_temps.append(group['temp'].values)
    w_workingday.append(group['workingday'].iloc[0])
    y_cnt.append(group['cnt'].sum())

# Create a DataFrame to store the results
data = pd.DataFrame({
    'date': dates,
    'x_temp': x_temps,
    'w_workingday': w_workingday,
    'y_total_cnt': y_cnt
})

data = data.sample(frac=1).reset_index(drop=True)

kf = KFold(n_splits=5,shuffle=True, random_state=42) 

x_temps = data['x_temp'].to_numpy()
w_workingday = data['w_workingday'].to_numpy()
y_cnt_origin = data['y_total_cnt'].to_numpy()
#y_cnt_origin = np.sqrt(y_cnt_origin)
y_cnt = (y_cnt_origin-np.mean(y_cnt_origin))/np.std(y_cnt_origin)

mean_pred_all = np.array([])
l2_mean_pred_all = np.array([])
y_test_all = np.array([])

lower_95 = np.array([])
lower_90 = np.array([])
lower_50 = np.array([])
upper_95 = np.array([])
upper_90 = np.array([])
upper_50 = np.array([])

l2_lower_95 = np.array([])
l2_lower_90 = np.array([])
l2_lower_50 = np.array([])
l2_upper_95 = np.array([])
l2_upper_90 = np.array([])
l2_upper_50 = np.array([])

for train_index, test_index in kf.split(data):

    train_index, valid_index = train_index[:len(train_index)//2], train_index[len(train_index)//2:]
    n_train = len(train_index)
    n_test = len(test_index)

    temp_train, temp_valid, temp_test = x_temps[train_index], x_temps[valid_index], x_temps[test_index]
    w_train, w_valid, w_test = w_workingday[train_index], w_workingday[valid_index], w_workingday[test_index]
    y_train, y_valid, y_test = y_cnt[train_index], y_cnt[valid_index], y_cnt[test_index]

    ############# L2 model ###################
    
    x_train = np.stack((np.stack(temp_train), np.repeat(w_train[:, np.newaxis], 24, axis=1)), axis=-1)
    x_valid = np.stack((np.stack(temp_valid), np.repeat(w_valid[:, np.newaxis], 24, axis=1)), axis=-1)
    x_test = np.stack((np.stack(temp_test), np.repeat(w_test[:, np.newaxis], 24, axis=1)), axis=-1)

    l2_model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=64,kernel_size=3,activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2,strides=2),
                tf.keras.layers.Conv1D(filters=64,kernel_size=3,activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=2,strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(1)
                ])
    
    # Compile the model
    l2_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mse'])

    # Train the model
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    l2_model.fit(x_train, y_train, epochs=2000, batch_size=500, validation_data=(x_valid, y_valid), callbacks=[early_stop])

    l2_mean_pred = l2_model.predict(np.array(x_test))[:,0]
    l2_mean_pred_all = np.append(l2_mean_pred_all, l2_mean_pred)
    y_test_all = np.append(y_test_all,y_test)

    res_train = np.array(y_train) - l2_model.predict(np.array(x_train))[:,0]
    t_train = np.sort(np.unique(y_train))
    l2_F = np.array([[sum((l2_mean_pred[k]+res_train)<t_train[j])/n_train for j in range(len(t_train))] for k in range(len(l2_mean_pred))])

    l2_lower_95 = np.append(l2_lower_95,[t_train[k] for k in np.argmax(l2_F > 0.025, axis=1)])
    l2_lower_90 = np.append(l2_lower_90,[t_train[k] for k in np.argmax(l2_F > 0.05, axis=1)])
    l2_lower_50 = np.append(l2_lower_50,[t_train[k] for k in np.argmax(l2_F > 0.25, axis=1)])
    l2_upper_95 = np.append(l2_upper_95,[t_train[k] for k in np.argmax(l2_F > 0.975, axis=1)])
    l2_upper_90 = np.append(l2_upper_90,[t_train[k] for k in np.argmax(l2_F > 0.95, axis=1)])
    l2_upper_50 = np.append(l2_upper_50,[t_train[k] for k in np.argmax(l2_F > 0.75, axis=1)])

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
        [64,64,10],
        [1, 64, 10],
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
    losshistory, train_state = model.train(iterations=20000,batch_size=2000)

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
    lower_50 = np.append(lower_50,[t_train[k] for k in np.argmax(F > 0.25, axis=1)])
    upper_95 = np.append(upper_95,[t_train[k] for k in np.argmax(F > 0.975, axis=1)])
    upper_90 = np.append(upper_90,[t_train[k] for k in np.argmax(F > 0.95, axis=1)])
    upper_50 = np.append(upper_50,[t_train[k] for k in np.argmax(F > 0.75, axis=1)])

    pmf = np.diff(F,axis=1,prepend=0)
    mean_pred = np.sum(pmf*t_test,axis=1)

    mean_pred_all = np.append(mean_pred_all, mean_pred)

    t_test = t_test*np.std(y_cnt_origin)+np.mean(y_cnt_origin)
    y_test = y_test*np.std(y_cnt_origin)+np.mean(y_cnt_origin)
    for j in range(len(F)):
        plt.plot(t_test,F[j])
        plt.axvline(x=y_test[j], color='r', linestyle='--', linewidth=2)
        #plt.show()

    
#plt.show()
mse = np.mean((mean_pred_all-y_test_all)**2)
medse = np.median((mean_pred_all-y_test_all)**2)
l2_mse = np.mean((l2_mean_pred_all-y_test_all)**2)
l2_medse = np.median((l2_mean_pred_all-y_test_all)**2)
print("mse:", mse, " medse:", medse, " l2_mse:", l2_mse, " l2_medse:", l2_medse)

cov_90 = np.mean((y_test_all >= lower_90) & (y_test_all <= upper_90))
cov_95 = np.mean((y_test_all >= lower_95) & (y_test_all <= upper_95))
cov_50 = np.mean((y_test_all >= lower_50) & (y_test_all <= upper_50))

l2_cov_90 = np.mean((y_test_all >= l2_lower_90) & (y_test_all <= l2_upper_90))
l2_cov_95 = np.mean((y_test_all >= l2_lower_95) & (y_test_all <= l2_upper_95))
l2_cov_50 = np.mean((y_test_all >= l2_lower_50) & (y_test_all <= l2_upper_50))

print("cov_50:", cov_50," cov_90:", cov_90, " cov_95:", cov_95)
print("l2_cov_50:", l2_cov_50," l2_cov_90:", l2_cov_90, " l2_cov_95:", l2_cov_95)

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