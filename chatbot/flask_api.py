def find(parent, a):
    if parent[a] == a:
        return a
    else:
        return find(parent, parent[a])


def union(a, b, parent):
    parenta = find(parent,a)
    parentb = find(parent,b)
    if parenta > parentb:
        parent[parenta] = parentb
    else:
        parent[parentb] = parenta
    print(f'a:{a},b:{b},parenta:{parenta},parentb:{parentb},parent:{parent}')

N_com = int(input())
N_edg = int(input())
answer = 0
edges = []
# visited=set()
parent = [i for i in range(N_com + 1)]
for _ in range(N_edg):
    a, b, D = map(int, input().split())
    edges.append((a, b, D))
edges = sorted(edges, key=lambda x: x[2])
print(f'edges:{edges}')
print(f'parent:{parent}')
for a, b, D in edges:
    if find(parent,a) != find(parent,b):
        union(a, b, parent)
        answer += D

        print(f'update:{a,b},{parent},{answer}')

print(answer)
# from flask import Flask, request
# from DBConnect import Database
# from bankinfo_insert import *
# from bankitem_insert import *
#
# db = Database()
#
# application = Flask(__name__)
#
#
# @application.route("/",methods=['GET'])
# def hello():
#     return "My name is hyunsoo!!"
#
#
# @application.route("/DB/apiinfoget",methods=['GET'])
# def apiinfoget():
#     global db
#     sql = 'truncate bankinfo.bankinfo'
#     db.execute(sql)
#     db.commit()
#     Total_bank_info = get_infos()
#     print(Total_bank_info)
#     insertbankinfo(Total_bank_info,db)
#     return "infoinsert"
#
# @application.route("/DB/apiitemget",methods=['GET'])
# def apiitemget():
#     sql = 'truncate bankinfo.bank_item'
#     db.execute(sql)
#     db.commit()
#     Total_bank_items = get_items()
#     print(Total_bank_items)
#     insertbank_item(Total_bank_items,db)
#     sql='update bankinfo.bank_item set max_limit=%s where max_limit=%s or max_limit=%s'
#     db.execute(sql,('1000000000','','0'))
#     db.commit()
#     return "iteminsert"
#
# @application.route("/DB/itemget",methods=['POST'])
# def itemget():
#     global db
#     if request.method == 'POST':
#         params=request.get_json()
#         money=int(params['money'])
#         intr_rate_type=params['intr_rate_type']
#         save_trm=int(params['save_trm'])
#         sql="select * from bankinfo.bank_item where max_limit>=%s and intr_rate_type=%s and save_trm=%s"
#         rows=db.executeAll(sql,(money,intr_rate_type,save_trm))
#         # max_intr=-1
#         # max_intr2=-1
#         # for r in rows:
#         #     intr_rate=float(r['intr_rate'])
#         #     intr_rate2=float(r['intr_rate2'])
#         #     intr_rate_type=r['intr_rate_type']
#         #     intr_money=interest_cal(money, save_trm, intr_rate, intr_rate_type)
#         #     intr_money2=interest_cal(money, save_trm, intr_rate2, intr_rate_type)
#         #     if max_intr<intr_money:
#         #         max_intr=intr_money
#         #         max_intr_item=r
#         #     if max_intr2<intr_money2:
#         #         max_intr2 = intr_money2
#         #         max_intr_item2 = r
#         return {"result":len(rows)}
#
# if __name__ == "__main__":
#     # print(sys.argv)
#     application.run(host='0.0.0.0', port=5000, debug=True)
