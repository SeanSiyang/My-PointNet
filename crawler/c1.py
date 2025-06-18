from urllib.request import urlopen

# 打开网址
resp = urlopen("http://www.baidu.com")
# 打印抓取到的内容
# print(resp.read().decode("utf-8"))

save_path = "E:/Codes/爬虫练习/baidu.html"

with open(save_path, mode="w", encoding="utf-8") as f:
    f.write(resp.read().decode("utf-8"))
    
print("over")

import requests

kw = input("请输入你要搜索的内容")
url = f'https://www.sogou.com/web?query={kw}'
headers = {
    "User-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWeb"
    "Kit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"
}

response = requests.get(url, headers=headers)
print(response.text)

response.close()