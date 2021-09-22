import requests, bs4
import pandas as pd
from lxml import html
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus, unquote

day_price_list=[]
api_key='cccb9b4c-90f3-4fb3-8e34-f029cc42b57c'
id='hyunsooklm961019@gmail.com'
year_list=['06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
for year in year_list:
    sample_url=f'''http://www.kamis.or.kr/service/price/xml.do?action=periodProductList&p_productclscode=01&p_startday=20{year}-01-01&p_endday=20{year}-12-31&p_itemcategorycode=200&p_itemcode=211&p_countrycode=1101&p_convert_kg_yn=N&p_cert_key={api_key}&p_cert_id={id}&p_returntype=xml'''
    print(f'{year} start!')
    response = requests.get(sample_url).text.encode('utf-8')
    xmlobj = bs4.BeautifulSoup(response, 'lxml-xml')
    items=xmlobj.find_all('item')
    print(len(items))
    start_month=1
    lst=[]
    for index,it in enumerate(items):
        # print(it['p_startday'])
        if it.regday==None:
            continue
        category=it.countyname.string
        day=it.regday.string
        year=it.yyyy.string
        day=str(day)[:2]+str(day)[3:]
        month=int(day[:2])
        price=it.price.string
        lst.append((year+day,price))
        if category!='평균':
            print(f'year:{year}')
            day_price_list.extend(lst)
            break
