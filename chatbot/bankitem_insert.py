import time
import requests
from bs4 import BeautifulSoup as bs
from collections import defaultdict
from DBConnect import *


#TODO 현재 MYSQL DB에 정보 모두 넣어놨음.

def interest_cal(money, save_trm, intr_rate, rsrv_type, Tax_type='normal'):
    interest = -1
    rsrv_type=rsrv_type.upper().strip()
    if rsrv_type == 'S':
        # 이자(단리): 월납입금 * n(n+1)/2 * r/12
        origin_interest = money * save_trm * (save_trm + 1) / 2 * (intr_rate / 12)
    elif rsrv_type == 'F':
        # 이자(복리): (월납입금 * (1 + r / 12) * ((1 + r / 12) ** n - 1) / (r / 12)) - (월납입금 * n)
        origin_interest = (money * (1 + intr_rate / 12) * ((1 + intr_rate / 12) ** save_trm - 1) / (intr_rate / 12)) - (money * save_trm)
    if Tax_type == 'normal':
        interest = origin_interest * 0.846
    if Tax_type == 'non_Tax':
        interest = origin_interest
    if Tax_type == 'tax preferential':
        interest = origin_interest * 0.905
    return int(interest)
#사용법: intr=interest_cal(100000,24,0.02,' s ')
def get_items():
    auth_key = 'b555824094d0be01126bc05694f89259'
    bank_code = '020000'
    savebank_code = '030300'
    loan = '030200'
    insurance = '050000'
    banking = [bank_code, savebank_code, loan, insurance]
    Total_bank_info = defaultdict(dict)
    num = 1
    api_start = time.time()
    for bankkind in banking:  # 권역별로 itemlist 따오기
        current_pageNo = 1
        while True:
            URL = f'http://finlife.fss.or.kr/finlifeapi/savingProductsSearch.xml?auth={auth_key}&topFinGrpNo={bankkind}&pageNo={current_pageNo}'
            response = requests.get(URL)
            if response.status_code == 200:
                soup = bs(response.content, 'lxml')
                products = soup.find_all('product')
                max_pageNo = int(soup.find('max_page_no').text)  # max_pageNo
                for p in products:
                    info = p.find('baseinfo')
                    ops = p.find_all('option')  # 해당 금융상품에서 option tag로 달린것들
                    for base in info.children:  # baseinfo태그값들
                        if base.name == None:
                            continue
                        Total_bank_info[num][
                            base.name.replace("\n", "")] = base.text  # base.name은 태그명, base.text는 태그 안 str
                    subm_day = Total_bank_info[num]['fin_co_subm_day']
                    try:
                        Total_bank_info[num]['fin_co_subm_day'] = subm_day[:4] + '-' + subm_day[4:6] + '-' + subm_day[6:8]
                    except:
                        pass
                    Total_bank_info[num]['option'] = []
                    for option in ops:
                        sub_opt = dict()
                        for child in option.children:
                            if child.name == None:
                                continue
                            sub_opt[child.name.replace("\n", "")] = child.text
                        Total_bank_info[num]['option'].append(sub_opt)
                    num += 1
                if current_pageNo < max_pageNo:  # 해당 권역코드의 금융상품 MAX Page에 맞춰 모두 가져오기.
                    current_pageNo += 1
                else:
                    break
            else:
                print('Http error occur!\n')
                break
    return Total_bank_info
def insertbank_item(Total_bank_info,db):
    for n,item in Total_bank_info.items():
        for op in item['option']:
            sql='''INSERT INTO bankinfo.bank_item(`fin_co_no`,`kor_co_nm`,`fin_prdt_cd`,`fin_prdt_nm`,`join_way`,
            `mtrt_int`,
            `spcl_cnd`,
            `join_deny`,
            `join_member`,
            `etc_note`,
            `max_limit`,
            `fin_co_subm_day`,
            `intr_rate_type`,
            `intr_rate_type_nm`,
            `rsrv_type`,
            `rsrv_type_nm`,
            `save_trm`,
            `intr_rate`,
            `intr_rate2`)
            VALUES
            (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
            '''
            ok=db.execute(sql,(item['fin_co_no'],item['kor_co_nm'],item['fin_prdt_cd'],item['fin_prdt_nm'],item['join_way'],
                item['mtrt_int'],item['spcl_cnd'],item['join_deny'],item['join_member'],item['etc_note'],item['max_limit'],
                item['fin_co_subm_day'],op['intr_rate_type'],op['intr_rate_type_nm'],op['rsrv_type'],op['rsrv_type_nm'],op['save_trm'],op['intr_rate'],
                op['intr_rate2']))
    db.commit()

if __name__ == "__main__":
    Total_bank_info=get_items()
    insertbank_item(Total_bank_info)
    print('Done')
