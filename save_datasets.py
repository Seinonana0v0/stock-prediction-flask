import pymysql
import tushare as ts
import datetime

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
    df = pro.daily(ts_code=stock, start_date='20190101', end_date=today)
    print(df)
    df = df[['trade_date', 'open', 'high', 'close', 'low', 'vol']]
    df = df.iloc[::-1]
    print(stock)
    df.to_csv('./datasets/' + stock + '.csv', index=False)

cursor.close()
