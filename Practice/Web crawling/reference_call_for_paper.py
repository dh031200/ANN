from bs4 import BeautifulSoup
import urllib.request
import re

url = 'https://jackietseng.github.io/conference_call_for_paper/conferences-with-ccf.html'
page = urllib.request.urlopen(url).read()
soup = BeautifulSoup(page, 'html.parser')

# print(soup.prettify())
max_len = 0
results = soup.find_all('tr')
table = []
for tr in results:
    row = []
    for td in tr.find_all('td'):
        contents = re.sub('[^0-9a-zA-Zㄱ-힕 ,:/.()-]', '', td.get_text())
        row.append(contents)
    if row[0] == 'B':
        table.append(row)

for i in table:
    print('|    ', end='')
    for j in i:
        # if len(i) > max_len:
        #     len = max_len
        print(j, end='    |    ')
    print('')
