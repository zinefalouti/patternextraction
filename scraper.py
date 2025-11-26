import requests
from bs4 import BeautifulSoup

url = 'https://forums.unrealengine.com/t/the-proper-way-of-releasing-updating-and-patching-your-ue4-games/154663'
response = requests.get(url)
html_content = response.text

soup = BeautifulSoup(html_content, 'html.parser')

li_tag = soup.find_all('li')

for l in li_tag:
    print(l)
