__author__ = 'Kevin Omar Ken'

#nltk module
import nltk
from nltk import regexp_tokenize
from nltk.book import FreqDist
from nltk.corpus import stopwords
from nltk.collocations import *
#sparce matrix modules
from scipy.sparse import  vstack
#from scipy.sparse import  hstack
#from scipy.sparse import coo_matrix

#sklearn modules
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle #random permutation
from sklearn import preprocessing # Normalize Sparces Matrix
from sklearn.feature_extraction.text import CountVectorizer # create at CSR from data
from sklearn.linear_model import LogisticRegression#Logistic Classifier
from sklearn import svm #SVM Clasiffier
from sklearn.naive_bayes import MultinomialNB#NAive BAIse
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#utilities
import numpy as np
import urllib.request
#time base modules
from workdays import workday
import calendar
from datetime import datetime
from time import sleep

def open_url(SYMBOL: str,START_MONTH: str, START_DAY: str,START_YEAR: str,END_YEAR: str,END_MONTH: str,END_DAY:str):
    print("http://ichart.finance.yahoo.com/table.csv?s="+SYMBOL+"&a="+START_MONTH+"&b="+START_DAY+"&c="+START_YEAR+"&d="+END_MONTH+"&e="+END_DAY+"&f="+END_YEAR+"&g=d")    
    raw_data = urllib.request.urlopen("http://ichart.finance.yahoo.com/table.csv?s="+SYMBOL+"&a="+START_MONTH+"&b="+START_DAY+"&c="+START_YEAR+"&d="+END_MONTH+"&e="+END_DAY+"&f="+END_YEAR+"&g=d", timeout = 100)
    return trim_data(raw_data.read().decode(encoding='UTF-8',errors='strict'))
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
#we're going to start by opening the file with the data
def generate_holidays_list(holiday_file):
    return [datetime(int(line[0:4]), int(line[4:6]), int(line[6:8])) for line in holiday_file]

e = open('holiday_list.txt','r',encoding="utf-8")
f = open('combined_data.txt', 'r',encoding="utf-8")
#some variables here...
bigram_measures = nltk.collocations.BigramAssocMeasures()
line = ''
article = ''
date = ''
symbol = ''
count = 1
current_tokens = []
label = []
features = []
opening_prices = []
count_vect = CountVectorizer(min_df = 1,stop_words = stopwords.words('english'))
holidays = generate_holidays_list(e)
pattern = '[a-z][a-z][a-z]*'
#now we're going to build the feature vectors
responce = input("Do you want to repopulate feature vectors?[Y/N]:")

for line in f:
        #pull out the date
#    print(line)
    if (line[0:7] == 'DATE = '):
        print("Processing article {}".format(count))
        count += 1
        date = line[7:]
#        print(date)
        date_object = datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]))
        #pull out the symbol
    elif (line[0:9] == 'SYMBOL = '):
        symbol = line[9:].strip('\n')
#        print(symbol)
    #use the information to populate the feature vector... (most of the work happens here)
    elif (line[0:12] == 'END OF ENTRY'):
        if (calendar.weekday(date_object.year,date_object.month,date_object.day) in [5,6]):
            date_object = workday(date_object,-12,holidays)#include NYSE holidays
        end_date = workday(date_object,11,holidays)
#        print(date_object)
#        print(end_date)
        if(responce in ['y',"Y"]):
            sleep(.45)
            data = open_url(symbol, str(date_object.month -1), str(date_object.day), str(date_object.year),str(end_date.year), str(end_date.month-1), str(end_date.day))
#            if(data[0] <= 0):
#                label.append(-1)
#            else:
#                    label.append(1)
            label.append(data[0])
            opening_prices.append(data[1][1])
        date = ''
        symbol = ''
        finder = BigramCollocationFinder.from_words(current_tokens)
        finder.apply_freq_filter(5)
        scored = sorted(finder.nbest(bigram_measures.raw_freq,100))
#        current_tokens = []
        for pair in scored:
            current_tokens += [str(pair[0])+str(pair[1])]   
#        print(current_tokens)
        features.append(current_tokens)
        current_tokens = []
    else:
        line = line.lower()
        tokens = regexp_tokenize(line, pattern)
        current_tokens +=[w for w in tokens if not w in stopwords.words('english')]
        
    #we're still building the article...
g = open('label.txt','r+', encoding="utf-8")
if responce in ['y',"Y"]:
    for classifier in label:
        g.write(str(classifier)+'\n')
else:
    for line in g:
        label.append(float(line.strip('\n')))
g.close()
big_feature_list = []
for feat in features:
    big_feature_list.append(' '.join(word for word in feat))
#h = open('pre_prossed_data','r+')
#
#for feat in big_feature_list:
#    h.write(feat+'\n')
#h.close()
#read_Data = open('pre_prossed_data','r')  
#for line in read_data:
#    big_feature_list.append(line.strip('\n'))
#read_data.close()
big_feature_list ,label = shuffle(big_feature_list, label, random_state=5)
label = label.tolist()
#data = [features, opening_prices]
corpus = []
for document in big_feature_list.tolist():
    corpus += document.split();
fdist = FreqDist(corpus)
#print(fdist.most_common(200))
#s = open('stopwords.txt' , 'w')
#for pair in fdist.most_common(500):
#    s.write(str(pair[0])+'\n')
#s.close()
xform = count_vect.fit_transform(big_feature_list)
xform = preprocessing.scale(xform, with_mean=False)
#tform = count_vect.fit_transform(opening_prices)
#xform = hstack([xform, tform])
#xform = coo_matrix.tocsr(xform)
f.close()
e.close()
#print(xform)
V = 12
M = count//V
accLR = []
accNB = []
accSVM = []
accBAGSVM = []
accBAGLR = []
accmajor = []
#tuned_param_SVM = [{'kernel' : ['linear'],'C':[1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]},{'kernel' : ['rbf'],'C':[1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4],'gamma':[1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]},{'kernel' : ['poly'], 'degree':[3,4,5],'C':[1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]}]
#tuned_param_LR = [{'penalty': ['l1','l2'],'C':[1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]}]
#tuned_param_MNB = [{'alpha':[1e-4,1e-3,1e-2,1e-1,1]}]
#
#scores = ['precision', 'recall']
#X_train, y_train = xform[:375], label[:375]
#x_test , y_test = xform[375:], label[375:]
#if __name__ == "__main__":
#    for score in scores:
#        print("# Tuning hyper-parameters for %s" % score)
#        print()
#
#        clfSVM = GridSearchCV(svm.SVC(), param_grid = tuned_param_SVM, cv=12, scoring=score , n_jobs = 4)
#        clfSVM.fit(X_train, y_train)
##    clfLR = GridSearchCV(LogisticRegression(C=1), param_grid  = tuned_param_LR, cv=12, scoring=score)
##    clfLR.fit(X_train, y_train)
##    clfNB = GridSearchCV( MultinomialNB(alpha = 0.01), tuned_param_MNB, cv=12, scoring=score)
##    clfNB.fit(X_train, y_train)
#        print("Best parameters set found on development set:")
#        print()
#        print(clfSVM.best_estimator_)
##    print()
##    print(clfLR.best_estimator_)
##    print()
##    print(clfNB.best_estimator_)
#        print()
#        print("Grid scores on development set:")
#        print()
#        for params, mean_score, scores in clfSVM.grid_scores_:
#            print("%0.3f (+/-%0.03f) for %r"
#              % (mean_score, scores.std() / 2, params))
#            print()
#
#        print("Detailed classification report:")
#        print()
#        print("The model is trained on the full development set.")
#        print("The scores are computed on the full evaluation set.")
#        print()
#        y_true = y_test
#        yhatSVM=clfSVM.predict(x_test)
##    yhatLR=clfLR.predict(x_test)
##    yhatNB=clfNB.predict(x_test)
#        print(classification_report(y_true, yhatSVM))
#        print()
##    print(classification_report(y_true, yhatLR))
##    print()
##    print(classification_report(y_true, yhatNB))
#        print()
mserl = []
for fold in range(V):
    first, last = fold*M, (fold+1)*M
    xtest , xtrain = xform[first:last], vstack([xform[:first],xform[last:]])
    ytest , ytrain = label[first:last], label[:first]+label[last:]
    clf = LinearRegression().fit(xtrain,ytrain)
    yhat = clf.predict(xtest)
    mserl.append(mean_squared_error(ytest,yhat))
print(mserl)
print(sum(mserl)/len(mserl))



#for fold in range(V):
#    first, last = fold*M, (fold+1)*M
#    xtest , xtrain = xform[first:last], vstack([xform[:first],xform[last:]])
#    ytest , ytrain = label[first:last], label[:first]+label[last:]
#    clfSVM = svm.LinearSVC(C =1).fit(xtrain, ytrain)
#    clfLR = LogisticRegression(penalty = 'l2', C = 1).fit(xtrain, ytrain)
#    clfNB = MultinomialNB(alpha = 0.01).fit(xtrain, ytrain)
#    clfBAGSVM = BaggingClassifier(base_estimator = svm.SVC(kernel = 'linear', C = 1), n_estimators = 50, max_samples = .75,max_features = .75).fit(xtrain,ytrain)    
#    clfBAGLR = BaggingClassifier(base_estimator = LogisticRegression(penalty = 'l2', C =1), n_estimators = 50, max_samples = .75,max_features = .75).fit(xtrain,ytrain)
#    yhatSVM=clfSVM.predict(xtest)
#    yhatLR=clfLR.predict(xtest)
#    yhatNB=clfNB.predict(xtest)
#    yhatBAGSVM = clfBAGSVM.predict(xtest)
#    yhatBAGLR = clfBAGLR.predict(xtest)
#    ymajority = [1 if (yhatSVM[i]+yhatLR[i]+yhatNB[i]+yhatBAGSVM[i]+yhatBAGLR[i] > 0) else -1 for i in range(len(ytest))]  
#    ymajority = numpy.asarray(ymajority)
#    accmajor.append(np.mean(ymajority==ytest))
#    accLR.append(np.mean(yhatLR==ytest))
#    accSVM.append(np.mean(yhatSVM==ytest))
#    accNB.append(np.mean(yhatNB==ytest))
#    accBAGSVM.append(np.mean(yhatBAGSVM==ytest))
#    accBAGLR.append(np.mean(yhatBAGLR==ytest))
#print('SVM ACCURACY:')
#print(accSVM)
#print(sum(accSVM)/len(accSVM))
#print('LR ACCURACY:')
#print(accLR)
#print(sum(accLR)/len(accLR))
#print('NB ACCURACY:')
#print(accNB)
#print(sum(accNB)/len(accNB))
#print('BAG SVM ACCURACY:')
#print(accBAGSVM)
#print(sum(accBAGSVM)/len(accBAGSVM))
#print('BAG LR ACCURACY:')
#print(accBAGLR)
#print(sum(accBAGLR)/len(accBAGLR))
#print('Majority ACCURACY:')
#print(accmajor)
#print(sum(accmajor)/len(accmajor))
