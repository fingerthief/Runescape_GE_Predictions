import json 
import csv
import os 
from urllib.request import urlopen
import pandas as pd
import requests
import time

solditems = requests.get('https://secure.runescape.com/m=itemdb_rs/api/graph/11720.json') # (your url)
data = solditems.json()
with open('data.json', 'w') as f:
    json.dump(data, f)

df = pd.read_json (r'data.json')
export_csv = df.to_csv (r'data.csv', index = None, header=True)

fr = open('data.csv', 'r')
reader = csv.reader(fr)
writer = csv.writer(open('data_final.csv', 'w'))
headers = next(reader)
headers.append("day")
writer.writerow(headers)
count = 1
for row in reader:
    row.append(str(count))
    writer.writerow(row)
    count += 1

fr.close()
os.remove('data.csv')

def csvAdder():
    fr = open('message.csv', 'r')
    reader = csv.reader(fr)
    writer = csv.writer(open('messages_final.csv', 'w'))
    headers = next(reader)
    headers.append("day")
    writer.writerow(headers)
    count = 1
    for row in reader:
        row.append(str(count))
        writer.writerow(row)
        count += 1

