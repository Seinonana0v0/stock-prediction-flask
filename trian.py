import math
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import time
import tensorflow as tf
import pydot


from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

def get_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_pred_eval_model(X_train, y_train, X_cv,y_cv,scaler, lstm_units=50,dropout_prob=0.5,optimizer='adam',epochs=1,batch_size=1):
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1],5)))
    model.add(Dropout(dropout_prob)) # Add dropout with a probability of 0.5
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_prob)) # Add dropout with a probability of 0.5
    model.add(Dense(30))
    
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    est_scaled = model.predict(X_cv)
    est = scaler.inverse_transform(est_scaled)
    
    
    y_cv_inv = scaler.inverse_transform(y_cv.reshape(len(y_cv),30))
    rmse = math.sqrt(mean_squared_error(y_cv_inv, est))
    mape = get_mape(y_cv_inv, est)
    
    return rmse, mape, est

def Train(stock_id,to_train):
    if stock_id.startswith('6'):
        stock_id = stock_id + '.sh'
    else:
        stock_id = stock_id + '.sz'

    csv_path = './datasets/'+stock_id+'.csv'
    df_ori = pd.read_csv(csv_path)
    df = df_ori[['close', 'open', 'high', 'low', 'vol']]
    mem_his_days = 60
    pre_days = 30
    length = len(df)-mem_his_days-pre_days
    scaler = MinMaxScaler(feature_range=(0, 1))
    num_cv=int(0.2*length)
    num_test=int(0.2*length)
    num_train = length - num_cv - num_test 

    df_train = df.loc[0:num_train+mem_his_days+pre_days-1]
    features_train = scaler.fit_transform(df_train)
    target_train = scaler.fit_transform(np.array(df_train[to_train]).reshape(-1,1))
    X_train = []
    y_train = []
    for each in range(len(features_train) - mem_his_days-pre_days):
        dataX = features_train[each:each + mem_his_days]
        datay = target_train[each + mem_his_days:each + mem_his_days + pre_days]
        
        X_train.append(dataX)
        y_train.append(datay)

    X_train = np.array(X_train)
    y_train = np.array(y_train) 

    df_cv = df.loc[num_train:num_train+num_cv+mem_his_days+pre_days-1]
    features_cv = scaler.fit_transform(df_cv)
    target_cv = scaler.fit_transform(np.array(df_cv[to_train]).reshape(-1,1))
    X_cv = []
    y_cv = []
    for each in range(len(features_cv) - mem_his_days-pre_days):
        dataX = features_cv[each:each + mem_his_days]
        datay = target_cv[each + mem_his_days:each + mem_his_days + pre_days]
        
        X_cv.append(dataX)
        y_cv.append(datay)

    X_cv = np.array(X_cv)
    y_cv = np.array(y_cv)

    df_test = df.loc[num_train+num_cv:]
    features_test = scaler.fit_transform(df_test)
    target_test = scaler.fit_transform(np.array(df_test[to_train]).reshape(-1,1))
    X_test = []
    y_test = []
    for each in range(len(features_test) - mem_his_days-pre_days):
        dataX = features_test[each:each + mem_his_days]
        datay = target_test[each + mem_his_days:each + mem_his_days + pre_days]

        X_test.append(dataX)
        y_test.append(datay)
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    df_train_cv=df.loc[0:num_train+num_cv+mem_his_days+pre_days-1]
    features_train_cv = scaler.fit_transform(df_train_cv)
    target_train_cv = scaler.fit_transform(np.array(df_train_cv[to_train]).reshape(-1,1))
    X_train_cv = []
    y_train_cv = []
    for each in range(len(features_train_cv) - mem_his_days-pre_days):
        dataX = features_train_cv[each:each + mem_his_days]
        datay = target_train_cv[each + mem_his_days:each + mem_his_days + pre_days]
        
        X_train_cv.append(dataX)
        y_train_cv.append(datay)

    X_train_cv = np.array(X_train_cv)
    y_train_cv = np.array(y_train_cv)

    lstm_units=50
    dropout_prob=0.5
    optimizer='adam'
    epochs=1
    batch_size=1

    param_label = 'epochs'
    param_list = [1,10,30,50,80,100]

    param2_label = 'batch_size'
    param2_list = [8, 16, 32, 64, 128]

    error_rate = {param_label: [], param2_label: [], 'rmse': [], 'mape_pct': []}
    tic = time.time()
    for param in tqdm_notebook(param_list):
        for param2 in tqdm_notebook(param2_list):
            rmse,mape,_=train_pred_eval_model(X_train,y_train,X_cv,y_cv,scaler,lstm_units=lstm_units,dropout_prob=dropout_prob,optimizer='adam',
                                            epochs=param, batch_size=param2)
            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate['rmse'].append(rmse)
            error_rate['mape_pct'].append(mape)
            
    error_rate = pd.DataFrame(error_rate)
    toc = time.time()
    print("Minutes taken = " + str((toc-tic)/60.0))
    error_rate 
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    epochs_opt = temp[param_label].values[0]
    batch_size_opt = temp[param2_label].values[0]

    param_label = 'lstm_units'
    param_list = [10, 50, 64, 128]

    param2_label = 'dropout_prob'
    param2_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    error_rate = {param_label: [], param2_label: [], 'rmse': [], 'mape_pct': []}
    tic = time.time()
    for param in tqdm_notebook(param_list):
        for param2 in tqdm_notebook(param2_list):
            rmse,mape,_=train_pred_eval_model(X_train,y_train,X_cv,y_cv,scaler,lstm_units=param,dropout_prob=param2,optimizer='adam',
                                            epochs=epochs_opt, batch_size=batch_size_opt)
            # Collect results
            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate['rmse'].append(rmse)
            error_rate['mape_pct'].append(mape)
            
    error_rate = pd.DataFrame(error_rate)
    toc = time.time()
    print("Minutes taken = " + str((toc-tic)/60.0))
    error_rate 

    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    lstm_units_opt = temp[param_label].values[0]
    dropout_prob_opt = temp[param2_label].values[0]

    param_label = 'optimizer'
    param_list = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']

    error_rate = {param_label: [], 'rmse': [], 'mape_pct': []}
    tic = time.time()
    for param in tqdm_notebook(param_list):
        rmse,mape,_=train_pred_eval_model(X_train,y_train,X_cv,y_cv,scaler,lstm_units=lstm_units_opt,dropout_prob=dropout_prob_opt,optimizer=param,
                                            epochs=epochs_opt, batch_size=batch_size_opt)
        error_rate[param_label].append(param)
        error_rate['rmse'].append(rmse)
        error_rate['mape_pct'].append(mape)
        
    error_rate = pd.DataFrame(error_rate)
    toc = time.time()
    print("Minutes taken = " + str((toc-tic)/60.0))
    error_rate 
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    optimizer_opt = temp[param_label].values[0]

    check_filepath = '.\\models\\' + stock_id + '\\' + f'model_' + to_train
    checkpoint = ModelCheckpoint(
        filepath=check_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True)

    model = Sequential()
    model.add(LSTM(units=lstm_units_opt, return_sequences=True, input_shape=(X_train.shape[1],5)))
    model.add(Dropout(dropout_prob_opt))
    model.add(LSTM(units=lstm_units_opt))
    model.add(Dropout(dropout_prob_opt))
    model.add(Dense(30))

    model.compile(loss='mean_squared_error', optimizer=optimizer_opt,metrics=['mape'])
    model.fit(X_train_cv, y_train_cv, epochs=epochs_opt, batch_size=batch_size_opt, validation_data=(X_test,y_test), 
    callbacks=[checkpoint])

