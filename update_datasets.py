import pymysql
import datetime
import tushare as ts

def UpdateDataSets():
    pro = ts.pro_api('e5cb98d953d7e9d92256d802a96e0779ad9181b7fe6dd9855b3e94b8')
    db = pymysql.connect(host='81.69.240.39', user='root', password='seinonanatr', database='db_stock')
    cursor = db.cursor()
    cursor.execute("SELECT id from t_stock ")
    results = cursor.fetchall()
    stock_list = []

    for row in results:
        stock_list.append(row[0])
    today = datetime.date.today().strftime('%Y%m%d')
    print(today)
    for stock in stock_list:
        if stock.startswith('6'):
            stock = stock + '.sh'
        else:
            stock = stock + '.sz'
        df = pro.daily(ts_code=stock, )
        df = df[['trade_date', 'open', 'high', 'close', 'low', 'vol']]
        print(df)
        df.to_csv('./datasets/' + stock + '.csv', mode='a', header=False, index=False)

    cursor.close()
