#functions 
import nltk
from nltk import regexp_tokenize
from nltk.collocations import *
from nltk.corpus import stopwords

#utilities
import urllib.request

#time based midules
from workdays import workday 
#'http://github.com/ogt/workdays' 
#All credit goesto author Odysseas Tsatalos.

import calendar
from datetime import datetime
from time import sleep

def open_url(SYMBOL: str,START_MONTH: str, START_DAY: str,START_YEAR: str,END_YEAR: str,END_MONTH: str,END_DAY:str):  
#    print("http://ichart.finance.yahoo.com/table.csv?s="+SYMBOL+"&a="+START_MONTH+"&b="+START_DAY+"&c="+START_YEAR+"&d="+END_MONTH+"&e="+END_DAY+"&f="+END_YEAR+"&g=d")
    raw_data = urllib.request.urlopen("http://ichart.finance.yahoo.com/table.csv?s="+SYMBOL+"&a="+START_MONTH+"&b="+START_DAY+"&c="+START_YEAR+"&d="+END_MONTH+"&e="+END_DAY+"&f="+END_YEAR+"&g=d", timeout = 100)
    return trim_data(raw_data.read().decode(encoding='UTF-8',errors='strict'))
def trim_data(data):
    parsed_data = data.split('\n')
    first_data = parsed_data[-2].split(',')
    second_data = parsed_data[1].split(',')
    delta = float(first_data[1]) - float(second_data[-1])
    return [delta, first_data, second_data]

def generate_holidays_list(holiday_file):
    return [datetime(int(line[0:4]), int(line[4:6]), int(line[6:8])) for line in holiday_file]
    
def process(responce)->list:
    #some variables here...
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    line = ''
    date = ''
    symbol = ''
    count = 1
    big_feature_list = []
    current_tokens = []
    label = []
    features = []
    pattern = '[a-z][a-z][a-z]*'
    if responce in ['y',"Y"]:
        #generate holiday list
    
        e = open('holiday_list.txt','r',encoding="utf-8")
        holidays = generate_holidays_list(e)
        e.close()
        
        f = open('combined_data.txt', 'r',encoding="utf-8")
        for line in f:
            #pull out the date
            if (line[0:7] == 'DATE = '):
                print("Processing article {}".format(count))
                count += 1
                date = line[7:]
                date_object = datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]))
            #pull out the symbol
            elif (line[0:9] == 'SYMBOL = '):
                symbol = line[9:].strip('\n')
            #use the information to populate the feature vector... (most of the work happens here)
            elif (line[0:12] == 'END OF ENTRY'):
                if (calendar.weekday(date_object.year,date_object.month,date_object.day) in [5,6]):
                    date_object = workday(date_object,-12,holidays)#include NYSE holidays
                end_date = workday(date_object,11,holidays)
                sleep(.4)#calls to api limited to one per 2.5ms
                data = open_url(symbol, str(date_object.month -1), str(date_object.day), str(date_object.year),str(end_date.year), str(end_date.month-1), str(end_date.day))
                if(data[0] <= 0):
                    label.append(-1)
                else:
                    label.append(1)
                date = ''
                symbol = ''
                finder = BigramCollocationFinder.from_words(current_tokens)
                finder.apply_freq_filter(5)
                scored = sorted(finder.nbest(bigram_measures.raw_freq,100))
                for pair in scored:
                    current_tokens += [str(pair[0])+str(pair[1])]   
                features.append(current_tokens)
                current_tokens = []
            else:
                line = line.lower()
                tokens = regexp_tokenize(line, pattern)
                current_tokens +=[w for w in tokens if not w in stopwords.words('english')]
            
        g = open('label.txt','w', encoding="utf-8")
        for classifier in label:
            g.write(str(classifier)+'\n')
        for feat in features:
            big_feature_list.append(' '.join(word for word in feat))
        h = open('pre_prossed_data.txt','w')
        for feat in big_feature_list:
            h.write(feat+'\n')
        f.close()
    else:
        g = open('label.txt','r+', encoding="utf-8")
        for line in g:
            label.append(int(line.strip('\n')))
        h = open('pre_prossed_data.txt','r+')
        for line in h:
            big_feature_list.append(line.strip('\n'))
    #close documnets:
    h.close()
    g.close()
    return big_feature_list,label