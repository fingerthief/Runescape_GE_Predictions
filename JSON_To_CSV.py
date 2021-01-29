import json 
import csv 
from urllib.request import urlopen
import pandas as pd

url = "https://secure.runescape.com/m=itemdb_rs/api/graph/11720.json"

response = urlopen(url)

data = json.loads(response.read())

# Opening JSON file and loading the data 
# into the variable data 
# with open('data.json') as json_file: 
#     data = json.load(json_file) 

arma_data = data['daily'] 

df = pd.read_json (r'arma.json')
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