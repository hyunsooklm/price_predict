import json
import urllib
import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait

import sys
# import requests
if sys.version_info[0]==3:
    from urllib.request import urlopen
else:
    from urllib import urlopen

# with urlopen("http://api.nobelprize.org/v1/prize.json") as url:
#     novel_prize_json_file=url.read()

simple_dict = {
    'abcadf1': '이승훈asfsdbc'
}

print(json.dumps(simple_dict,indent=4,ensure_ascii=False))
print(json.dumps(simple_dict,indent=4,ensure_ascii=True))


#json 쓰기 -> with open(~~~file,w) as json_file:
#                 json.dump() ->json파일 써짐

#python->json파일 직렬화 (json.dumps(dictionary,indent=4) ->str형태임
#
#loads->string->json형태로
#dumps->dictionary->json형태로 (str로 찍힘)
#json->python dictionary
#