from urllib.request import urlopen

# 打开网址
resp = urlopen("http://www.baidu.com")
# 打印抓取到的内容
# print(resp.read().decode("utf-8"))

save_path = "E:/Codes/爬虫练习/baidu.html"

with open(save_path, mode="w", encoding="utf-8") as f:
    f.write(resp.read().decode("utf-8"))
    
print("over")

# 1.8 requests 模块入门

import requests

# 抓取搜狗搜索内容
kw = input("请输入你要搜索的内容")
url = f'https://www.sogou.com/web?query={kw}'
headers = {
    "User-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWeb"
    "Kit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"
}

response = requests.get(url, headers=headers)   # 处理反爬
print(response.text)    # 获取页面源代码

response.close()

# 抓取百度翻译数据

url = "https://fanyi.baidu.com/sug"
kw = input("请输入你要翻译的英文单词：")
dic = {
    "kw": kw
}

# sug这个url通过post方式提交，所以需要模拟post请求
# 发送post请求，发送的数据必须在字典中，通过data参数进行传递
response = requests.post(url, data=dic)

# 返回值是json
resp_json = response.json()
print(resp_json)
print(resp_json['data'][0]['v'])

response.close()

# 抓取豆瓣电影
response = requests.get(url="https://movie.douban.com/")
print(response.headers)
# print(response.text)
if response.text:
    print(response.text)
response.close()

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"
}

response = requests.get(url="https://movie.douban.com/",
                        headers=headers)

print(response.headers)
print(response.text)

response.close()