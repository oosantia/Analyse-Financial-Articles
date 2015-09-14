# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 21:20:52 2015

@author: L
"""
import csv

SYMBOL = 'AAPL'
START_MONTH = '02'
START_DAY = '04'
START_YEAR = '2009'
END_MONTH = '03'
END_DAY = '06'
END_YEAR = '2015'

start = []
end = []

with open(SYMBOL +'.csv', 'rt') as csvfile:
#with open('experiment.csv', 'rt') as csvfile:
    mycsv = csv.reader(csvfile)
    mycsv = list(mycsv)
    for x, row in enumerate(mycsv):
        for field in row:
             if field == START_YEAR + '-' + START_MONTH + '-' + START_DAY:
                    start.extend(mycsv[x])
             elif field == END_YEAR + '-' + END_MONTH + '-' + END_DAY:
                    end.extend(mycsv[x])
                
delta = float(start[1]) - float(end[-1])



def trim_data(data):
#    print(data)
    parsed_data = data.split('\n')
#    print(parsed_data)
    first_data = parsed_data[-2].split(',')
    second_data = parsed_data[1].split(',')
#    print(first_data)
#    print(second_data)
    delta = float(first_data[1]) - float(second_data[-1])
    return [delta, first_data, second_data]
    
    