from flask import Flask, request
from DBConnect import Database
from bankinfo_insert import *
from bankitem_insert import *

db = Database()

application = Flask(__name__)


@application.route("/",methods=['GET'])
def hello():
    return "My name is hyunsoo!!"


@application.route("/DB/apiinfoget",methods=['GET'])
def apiinfoget():
    sql = 'truncate bankinfo.bankinfo'
    db.execute(sql)
    db.commit()
    Total_bank_info = get_infos()
    insertbankinfo(Total_bank_info,db)
    return "infoinsert"

@application.route("/DB/apiitemget",methods=['GET'])
def apiitemget():
    sql = 'truncate bankinfo.bank_item'
    db.execute(sql)
    db.commit()
    Total_bank_items = get_items()
    insertbank_item(Total_bank_items,db)
    return "iteminsert"


@application.route("/DB/itemget",methods=['POST'])
def getitem():
    if request.method == 'POST':
        params=request.get_json()
        money=params['money']
        intr_rate_type=params['intr_rate_type']
        save_trm=params['save_trm']
        sql="select * from bankinfo.bank_item where max_limit>=%s and intr_rate_type=%s and save_trm=%s"
        rows=db.executeAll(sql,(money,intr_rate_type,save_trm))
        for r in rows:
            print(r['max_limit'], r['intr_rate_type'],r['save_trm'])
        return 'done'

@application.route("/DB", methods=['POST'])
def select():
    if request.method == 'POST':
        params = request.get_json()
        name = params['name']
        sql = "select * from test.book where name=%s;"
        row = db.executeAll(sql, name)
        for r in row:
            print(r['num'] + 100)
        return {"result": row}
    else:
        return None


@application.route("/DB/insert", methods=['POST'])
def insert():
    if request.method == 'POST':
        print('post')
        # db = Database()
        print(request.is_json)
        params = request.get_json()
        book = params["book"]
        name = params["name"]
        num = int(params["num"])
        print(f'book{book} ,name:{name}, num:{num}')
        db = Database()
        sql = f'INSERT INTO `test`.`book`(`book`,`name`,`num`) VALUES(%s,%s,%s);'
        ok = db.execute(sql, (book, name, num))
        db.commit()
        db.close()
    else:
        return None


if __name__ == "__main__":
    # print(sys.argv)
    application.run(host='0.0.0.0', port=5000, debug=True)
