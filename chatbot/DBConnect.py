import pymysql

db=pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='kimhs1019@',
    db='bankinfo',
    charset='utf8'
)