import json 
import csv
import os 
from urllib.request import urlopen
import pandas as pd
import requests
import time

def DownloadJSON(item):
    solditems = requests.get('https://secure.runescape.com/m=itemdb_rs/api/graph/'+item +'.json') # (your url)
    data = solditems.json()
    with open('data.json', 'w') as f:
        json.dump(data, f)

def ConvertJSONtoCSV(filename):
    df = pd.read_json(filename)
    df.to_csv('data.csv', index=None, header=True)

def AddDataToCSV(reader):
    file = open('data_'+item+'.csv', 'w')
    writer = csv.writer(file)
    headers = next(reader)
    headers.append("day")
    writer.writerow(headers)
    count = 1
    for row in reader:
        #time.sleep(0.1)
        r = row
        r.append(str(count))
        print (str(row))
        writer.writerow(r)
        count += 1
    file.flush()
    file.close()

for fname in os.listdir('.'):
    if fname.startswith("data_"):
        os.remove(os.path.join('.', fname))

with open('items.txt', 'r') as items:
    for item in items:
        item = item.replace('\n','')
        print (item)
        DownloadJSON(item)
        ConvertJSONtoCSV('data.json')
        fr = open('data.csv', 'r')
        reader = csv.reader(fr)
        AddDataToCSV(reader)
        fr.close()
        os.remove('data.csv')