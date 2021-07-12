import time
import requests
import json
from bs4 import BeautifulSoup as bs
from collections import defaultdict

auth_key = 'b555824094d0be01126bc05694f89259'
bank_code = '020000'
savebank_code = '030300'
loan = '030200'
insurance = '050000'
banking = [bank_code, savebank_code, loan, insurance]
Total_bank_info = defaultdict(dict)
num=1
local_num=['02','031','032','033','041']
# URL=f'http://finlife.fss.or.kr/finlifeapi/companySearch.xml?auth={발급받은 인증키}&topFinGrpNo=020000&pageNo=1'
for bankkind in banking:  # 권역별로 itemlist 따오기
    current_pageNo = 1
    while True:
        URL = f'http://finlife.fss.or.kr/finlifeapi/companySearch.xml?auth={auth_key}&topFinGrpNo={bankkind}&pageNo={current_pageNo}'
        response = requests.get(URL)
        if response.status_code == 200:
            soup = bs(response.content, 'lxml')
            products = soup.find_all('product')
            max_pageNo = int(soup.find('max_page_no').text)  # max_pageNo
            total_count = int(soup.find('total_count').text)  # max_pageNo
            for p in products:
                info = p.find('baseinfo')
                for base in info.children:  # baseinfo태그값들
                    if base.name == None:
                        continue
                    Total_bank_info[num][base.name.replace("\n", "")] = base.text  # base.name은 태그명, base.text는 태그 안 str
                num+=1
            if current_pageNo < max_pageNo:  # 해당 권역코드의 금융상품 MAX Page에 맞춰 모두 가져오기.
                current_pageNo += 1
            else:
                break
        else:
            print('Http error occur!\n')
            break
    # print(len(Total_bank_info.keys()))
    for key,value in Total_bank_info.items():
        phone_number=Total_bank_info[key]['cal_tel']
        if len(phone_number)==8:
            phone_number=phone_number[:4]+'-'+phone_number[4:]
            print(f'AAA:{phone_number}')
        else:
            print(f'BBB:{phone_number}')