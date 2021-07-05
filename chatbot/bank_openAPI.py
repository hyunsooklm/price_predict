import requests
import json
import numpy as np
import pandas as pd
auth_key='b555824094d0be01126bc05694f89259'
bank_code='020000'
savebank_code='030300'
loan='030200'
insurance='050000'
total_item=[]
banklist=[bank_code,savebank_code,loan,insurance]
for fincode in banklist:
    current_page = 1
    item_list = []
    while True:
        URL=f'http://finlife.fss.or.kr/finlifeapi/savingProductsSearch.json?auth={auth_key}&topFinGrpNo={fincode}&pageNo={current_page}'
        print(URL)
        response=requests.get(URL)
        if response.status_code==200:
            data = response.json()
            # print(data['result'])
            max_page=int(data['result']['max_page_no'])
            n_item_list=data['result']['baseList']
            #print(n_item_list)
            item_list.extend(n_item_list)
            print(len(item_list))
            if current_page<max_page:
                current_page+=1
            else:
                break
    total_item.extend(item_list)
print(len(total_item))
for item in total_item:
    print(item)