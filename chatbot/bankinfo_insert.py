import requests
from bs4 import BeautifulSoup as bs
from collections import defaultdict
from DBConnect import *


def parse_PhoneNumber(string):
    global local_num
    string=string.replace('-','')
    if len(string)==8:
        phone_number=string[:4]+'-'+string[4:]
        if not string.startswith('1'):
            return '02-'+phone_number
        return phone_number
    else:
        if string.startswith('02'):
            return '02'+'-'+string[2:6]+'-'+string[6:]
        else:
            return string[:3]+'-'+string[3:6]+'-'+string[6:]


def insertbankinfo(Total_bank_info,db):
    # global db
    print(Total_bank_info)
    for key, value in Total_bank_info.items():
        cal_tel = parse_PhoneNumber(Total_bank_info[key]['cal_tel'])
        dcls_chrg_man = Total_bank_info[key]['dcls_chrg_man']
        fin_co_no = Total_bank_info[key]['fin_co_no']
        homp_url = Total_bank_info[key]['homp_url']
        kor_co_nm = Total_bank_info[key]['kor_co_nm']
        sql = '''INSERT INTO bankinfo.bankinfo(`fin_co_no`,`kor_co_nm`,`dcls_chrg_man`,`homp_url`,`cal_tel`)
        VALUES(%s,%s,%s,%s,%s);'''

        ok = db.execute(sql, (fin_co_no, kor_co_nm, dcls_chrg_man, homp_url, cal_tel))

    db.commit()
def get_infos():
    auth_key = 'b555824094d0be01126bc05694f89259'
    bank_code = '020000'
    savebank_code = '030300'
    loan = '030200'
    insurance = '050000'
    banking = [bank_code, savebank_code, loan, insurance]
    Total_bank_info = defaultdict(dict)
    num = 1
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
                        Total_bank_info[num][
                            base.name.replace("\n", "")] = base.text  # base.name은 태그명, base.text는 태그 안 str
                    num += 1
                if current_pageNo < max_pageNo:  # 해당 권역코드의 금융상품 MAX Page에 맞춰 모두 가져오기.
                    current_pageNo += 1
                else:
                    break
            else:
                print('Http error occur!\n')
                break
    return Total_bank_info
if __name__=="__main__":
    Total_bank_info=get_items()
    insertbankinfo(Total_bank_info)
    print('done')