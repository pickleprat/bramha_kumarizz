import requests 
import json 

question = input("Enter your query: ")  
headers = {"Content-Type": "application/json"} 


message = {"message": question}  


with requests.post("http://localhost:4000/generate/", 
                   headers=headers, stream = True, data=json.dumps(message)) as r:
    for chunk in r.iter_content(16): 
        print(chunk.decode("UTF-8"), end="")
