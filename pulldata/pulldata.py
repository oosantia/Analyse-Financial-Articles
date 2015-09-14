# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:33:57 2015

@author: L
"""
#import urllib.request
from urllib import request

import csv


from time import sleep

SYMBOL = ''
START_MONTH = '01'
START_DAY = '01'
START_YEAR = '2009'
END_MONTH = '03'
END_DAY = '06'
END_YEAR = '2015'

with open('symbols.csv', 'rt') as csvfile:
#with open('experiment.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        SYMBOL = row[0]
        response = request.urlopen("http://ichart.finance.yahoo.com/table.csv?s="+SYMBOL+"&a="+START_MONTH+"&b="+START_DAY+"&c="+START_YEAR+"&d="+END_MONTH+"&e="+END_DAY+"&f="+END_YEAR+"&g=d", timeout = 100)
        
        csv = response.read()
        
        # Save the string to a file
        csvstr = str(csv).strip("b'")
        
        lines = csvstr.split("\\n")
        f = open(SYMBOL + '.csv', "w")
        for line in lines:
           f.write(line + "\n")
        f.close()
        sleep(.45)
'''
def open_url(SYMBOL: str,START_MONTH: str, START_DAY: str,START_YEAR: str,END_YEAR: str,END_MONTH: str,END_DAY:str):
    print("http://ichart.finance.yahoo.com/table.csv?s="+SYMBOL+"&a="+START_MONTH+"&b="+START_DAY+"&c="+START_YEAR+"&d="+END_MONTH+"&e="+END_DAY+"&f="+END_YEAR+"&g=d")    
    raw_data = urllib.request.urlopen("http://ichart.finance.yahoo.com/table.csv?s="+SYMBOL+"&a="+START_MONTH+"&b="+START_DAY+"&c="+START_YEAR+"&d="+END_MONTH+"&e="+END_DAY+"&f="+END_YEAR+"&g=d", timeout = 100)


csvwriter.writerows(rows)
'''


'''
csvfile = open(SYMBOL + '.csv', 'w', newline='')
writer = csv.writer(csvfile)
#writer.writerow(['ID', 'Prediction'])


writer.writerows(urllib.request.urlopen("http://ichart.finance.yahoo.com/table.csv?s="+SYMBOL+"&a="+START_MONTH+"&b="+START_DAY+"&c="+START_YEAR+"&d="+END_MONTH+"&e="+END_DAY+"&f="+END_YEAR+"&g=d", timeout = 100))


#for q in range(len(merp)):
#    writer.writerow([q+1, merp[q]])
csvfile.close()
'''


'''
response = request.urlopen("http://ichart.finance.yahoo.com/table.csv?s="+SYMBOL+"&a="+START_MONTH+"&b="+START_DAY+"&c="+START_YEAR+"&d="+END_MONTH+"&e="+END_DAY+"&f="+END_YEAR+"&g=d", timeout = 100)

csv = response.read()

# Save the string to a file
csvstr = str(csv).strip("b'")

lines = csvstr.split("\\n")
f = open(SYMBOL + '.csv', "w")
for line in lines:
   f.write(line + "\n")
f.close()
'''