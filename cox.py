from lifelines import CoxTimeVaryingFitter
import pandas as pd
import numpy as np

def reformat_data(x,y=None):
    n = x[0].shape[0] // (x[0].shape[1]-2)
    m = x[0].shape[1]-2
    id_list = []
    x_list = []
    w_list = []
    z_list = []
    start_list = []
    stop_list = []
    fail_list = []
    for i in range(n):
        id_list += [i]*m
        x_list += list(x[0][(i+1)*m-1][:-2])
        w_list += [x[0][(i+1)*m-1][-2]] * m
        z_list += [x[0][(i+1)*m-1][-1]] * m
        start_list += [0]+list(x[1][:m-1].squeeze())
        stop_list += list(x[1][:m].squeeze())
        if y is not None:
            fail_list += list(y[0][i*m:(i+1)*m].squeeze())

    if y is not None:
        df = pd.DataFrame({'id':id_list,'fail':fail_list,'X':x_list,'W':w_list,'Z':z_list,'start':start_list,'stop':stop_list})
    else:
        df = pd.DataFrame({'id':id_list,'X':x_list,'W':w_list,'Z':z_list,'start':start_list,'stop':stop_list})     
    return df

    

def cox_pred(data,test_data):
    ctv = CoxTimeVaryingFitter()
    ctv.fit(data,id_col="id",event_col="fail",start_col="start",stop_col="stop",show_progress=True)
    beta = ctv.params_
    base_ch = ctv.baseline_cumulative_hazard_/np.exp(np.mean(data['X']*beta['X']+data['W']*beta['W']+data['Z']*beta['Z']))
    base_ch = base_ch.diff().fillna(base_ch)
    test_data['pred_hazard'] = 0
    for t in base_ch.index:
        location = (test_data['start']<t) & (test_data['stop']>=t)
        test_data.loc[location,'pred_hazard'] += \
            base_ch['baseline hazard'][t]*np.exp(test_data.loc[location,'X']*beta['X']+
            test_data.loc[location,'W']*beta['W']+test_data.loc[location,'Z']*beta['Z'])
    test_data['pred_survival_cox'] = test_data.groupby(['id'])['pred_hazard'].cumsum()
    test_data['pred_survival_cox'] = np.exp(-test_data['pred_survival_cox'])
    del test_data['pred_hazard']
    return test_data, beta