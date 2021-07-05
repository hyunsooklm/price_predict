from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time


URL='http://finance.moneta.co.kr/saving/bestIntCat02List.jsp?accu_meth=1&kid_edu_fg=&join_term=1Y&rgn_sido=&rgn_gugun=&org_grp_cd=BK&saving_amt=1000000'
driver=webdriver.Chrome('chromedriver')
'join_term=1Y&rgn_sido=&rgn_gugun=&org_grp_cd=BK&saving_amt=1000000'
driver.implicitly_wait(3)

#페이지이동
driver.get('https://www.naver.com')

#검색창을 찾고, 고슴도리 입력
search=driver.find_element_by_css_selector('#query')
search.send_keys('입력')
#search.send_keys(Keys.ENTER)
btn=driver.find_element_by_css_selector('#search_btn')
btn.click()
time.sleep(5)
driver.quit()