import csv
import glob
import os

outputfilename = 'data_joined.csv'
try:
    os.remove(outputfilename)
except:
    print('')

itemdata = glob.glob('*.csv')
headers = []
headers.append('day')

#open all of the files.
readers = []
for p in itemdata:
    f = open(p, 'r')
    readers.append(f)
#read the cell from all of the files at once.
content = {}
for reader in readers:
    content[reader.name] = []
    for row in reader:
        content[reader.name].append(row)

outputfile = open(outputfilename, 'w')
writer = csv.writer(outputfile)

#write headers to new file.
for prediction in itemdata:
    itemID = str(prediction
        .replace('data_', '')
        .replace('.csv', ''))
    headers.append(itemID + ' daily')
    headers.append(itemID + ' avg')
writer.writerow(headers)

#build a row from all of the cells
row = []
for i in range(1,180):
    row.append(i-1)
    for k in content:
        lines = content[k]
        daily = lines[i].split(',')[0]
        average = lines[i].split(',')[1]
        row.append(daily)
        row.append(average)
    
    
    #write row to new file.
    writer.writerow(row)
    #clear row
    row = []

outputfile.flush()
outputfile.close()
