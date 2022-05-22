from flask import Flask
from flask import make_response
from predict import Predict
import pandas as pd
import json
from update_datasets import UpdateDataSets
from trian import Train

app = Flask(__name__)

@app.route('/predict/<stock_id>/<to_predict>')
def predict(stock_id, to_predict):
    y = Predict(stock_id, to_predict)
    if stock_id.startswith('6'):
        stock_id = stock_id + '.sh'
    else:
        stock_id = stock_id + '.sz'
    df = pd.read_csv('./datasets/' + stock_id + '.csv')
    last_value = df[to_predict].tail(1).values.tolist()
    y.insert(0, last_value[0])
    res_y = [round(x, 2) for x in y]
    json_data = json.dumps(res_y)
    response = make_response(json_data)
    return response


@app.route('/train/<stock_id>/<to_train>')
def train(stock_id, to_train):
    try:
        Train(stock_id, to_train)
        return 'success'
    except:
        return 'error'


@app.route('/updatedatasets')
def update():
    UpdateDataSets()
