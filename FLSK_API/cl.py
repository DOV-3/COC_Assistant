import requests
user_info = {'question': "介绍一下模组奥斯里斯的陨落", 'history': []}
r = requests.post("http://127.0.0.1:5001/langchain", data=user_info)  
print(r.text)
["\n\n","\n"," ",""]