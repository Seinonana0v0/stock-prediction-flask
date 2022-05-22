import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

scaler = MinMaxScaler(feature_range=(0, 1))
def round_up(value):
    return round(value * 100) / 100.0


def Predict(stock_id, to_predict):
    if stock_id.startswith('6'):
        stock_id = stock_id + '.sh'
    else:
        stock_id = stock_id + '.sz'

    model = load_model('./models/' + stock_id+'/model_'+to_predict)
    csv_path = './datasets/' + stock_id + '.csv'
    df = pd.read_csv(csv_path)
    df = df[['close', 'open', 'high', 'low', 'vol']]
    print(df)
  
    sca_X = scaler.fit_transform(df)
    sca_y = scaler.fit_transform(np.array(df[to_predict]).reshape(-1,1))
    X = []

    for each in range(len(sca_X) - 60):
        dataX = sca_X[each:each + 60]
        X.append(dataX)

    X = np.array(X)

  
    y = model.predict(X)
    y_inv = scaler.inverse_transform(y)
    return y_inv[-1].tolist()

