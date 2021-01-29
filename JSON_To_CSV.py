import json 
import csv 
from urllib.request import urlopen
import pandas as pd
import requests

solditems = requests.get('https://secure.runescape.com/m=itemdb_rs/api/graph/11720.json') # (your url)
data = solditems.json()
with open('data.json', 'w') as f:
    json.dump(data, f)

df = pd.read_json (r'data.json')
export_csv = df.to_csv (r'armadyl.csv', index = None, header=True)

reader = csv.reader(open('armadyl.csv', 'r'))
writer = csv.writer(open('armadyl_final.csv', 'w'))
headers = next(reader)
headers.append("day")
writer.writerow(headers)
count = 1
for row in reader:
    row.append(str(count))
    writer.writerow(row)
    count += 1