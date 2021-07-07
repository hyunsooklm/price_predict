import time
import requests
import json
from bs4 import BeautifulSoup as bs
from collections import defaultdict
def from_user():
    money = ""
    while True:
        rsrv_type = input('정액 적립식:S 자유 적립식:F 입력하세요:')
        rsrv_type = rsrv_type.upper()
        if rsrv_type == 'S' or rsrv_type == 'F':
            break
        print('적금 방식 재선택 요망')
    if rsrv_type == 'S':
        while True:
            try:
                money = int(input("월 저축금액: "))
                if money < 0:
                    print("돈을 적으라고")
                else:
                    break
            except ValueError:
                print("돈을 적으라고")
    return rsrv_type, money
def interest_cal(money, save_trm, intr_rate, rsrv_type, Tax_type='normal'):
    interest = -1
    rsrv_type=rsrv_type.upper().strip()
    if rsrv_type == 'S':
        # 이자(단리): 월납입금 * n(n+1)/2 * r/12
        origin_interest = money * save_trm * (save_trm + 1) / 2 * (intr_rate / 12)
    elif rsrv_type == 'F':
        # result = (m * (1 + r / 12) * ((1 + r / 12) ** n - 1) / (r / 12)) - (m * n)
        origin_interest = (money * (1 + intr_rate / 12) * ((1 + intr_rate / 12) ** save_trm - 1) / (intr_rate / 12)) - (money * save_trm)
    if Tax_type == 'normal':
        interest = origin_interest * 0.846
    if Tax_type == 'non_Tax':
        interest = origin_interest
    return int(interest)
#사용법: intr=interest_cal(100000,24,0.02,' s ')
if __name__ == "__main__":
    intr = interest_cal(100000, 24, 0.02, 'S')
    print('안녕하세요. 적금서비스입니다.')
    rsrv_type,money=from_user()
    print(rsrv_type,money)
    auth_key='b555824094d0be01126bc05694f89259'
    bank_code='020000'
    savebank_code='030300'
    loan='030200'
    insurance='050000'
    banking=[bank_code,savebank_code,loan,insurance]
    Total_bank_info=defaultdict(dict)
    num=1
    api_start=time.time()
    for bankkind in banking:    #권역별로 itemlist 따오기
        current_pageNo=1
        while True:
            URL=f'http://finlife.fss.or.kr/finlifeapi/savingProductsSearch.xml?auth={auth_key}&topFinGrpNo={bankkind}&pageNo={current_pageNo}'
            response=requests.get(URL)
            if response.status_code==200:
                soup=bs(response.content,'lxml')
                products=soup.find_all('product')
                max_pageNo=int(soup.find('max_page_no').text)     #max_pageNo
                for p in products:
                    info=p.find('baseinfo')
                    ops=p.find_all('option')         #해당 금융상품에서 option tag로 달린것들
                    for base in info.children:       #baseinfo태그값들
                        if base.name==None:
                            continue
                        Total_bank_info[num][base.name.replace("\n","")]=base.text #base.name은 태그명, base.text는 태그 안 str
                    Total_bank_info[num]['option']=[]
                    for option in ops:
                        sub_opt=dict()
                        for child in option.children:
                            if child.name==None:
                                continue
                            sub_opt[child.name.replace("\n","")]=child.text
                        Total_bank_info[num]['option'].append(sub_opt)
                    num+=1
                if current_pageNo<max_pageNo:     #해당 권역코드의 금융상품 MAX Page에 맞춰 모두 가져오기.
                    current_pageNo+=1
                else:
                    break
            else:
                print('Http error occur!\n')
                break
    api_end=time.time()
    print(f'api따오는데 걸리는 시간:{api_end-api_start} 초')
    print(Total_bank_info[1]['spcl_cnd'])
    # for n in Total_bank_info.keys():
    #     print(len(Total_bank_info[n]['option']))