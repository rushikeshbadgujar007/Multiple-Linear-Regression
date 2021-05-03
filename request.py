import requests

url = 'http://localhost:5000/predict'
r = requests.post(url,json={'R&D Spend':1000.23, 'Administration':124153.04, 'Marketing Spend':1903.93 })
print(r.json())