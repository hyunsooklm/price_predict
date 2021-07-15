import pymysql

class Database():
    def __init__(self):
        self.db = pymysql.connect(host='127.0.0.1',
                                  port=3306,
                                  user='root',
                                  password='kimhs1019@',
                                  db='bankinfo',
                                  charset='utf8')
        self.cursor = self.db.cursor(pymysql.cursors.DictCursor)

    def execute(self, query, args={}):      #insert할 때
        self.cursor.execute(query, args)

    def executeOne(self, query, args={}):   #select 하나 가져올 때
        self.cursor.execute(query, args)
        row = self.cursor.fetchone()
        return row

    def executeAll(self, query, args={}):   #select 여러개 가져올 때
        self.cursor.execute(query, args)
        row = self.cursor.fetchall()
        return row

    def commit(self):
        self.db.commit()
    def close(self):
        self.db.close()