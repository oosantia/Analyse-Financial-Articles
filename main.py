__author__ = 'Kevin Omar Ken'

#imports

#nltk module
#from nltk.book import FreqDist
from nltk.corpus import stopwords

#sparce matrix modules
from scipy.sparse import  vstack

#sklearn modules
from sklearn.utils import shuffle #random permutation
from sklearn import preprocessing # Normalize Sparces Matrix
from sklearn.feature_extraction.text import CountVectorizer # create at CSR from data
from sklearn.linear_model import LogisticRegression#Logistic Classifier
from sklearn import svm #SVM Clasiffier
from sklearn.naive_bayes import MultinomialNB#Naive BAIse
from sklearn.ensemble import BaggingClassifier

#utilities
import numpy as np
import function

if __name__ == "__main__":
    responce = input("Do you want to repopulate feature vectors?[Y/N]:")
    big_feature_list ,label = function.process(responce)
    
    #shuffle data
    big_feature_list ,label = shuffle(big_feature_list, label, random_state=5)
    label = label.tolist()

    #build stopword list Not used
#    for document in big_feature_list.tolist():
#    corpus += document.split();
#    fdist = FreqDist(corpus)
#    print(fdist.most_common(200))
#    s = open('stopwords.txt' , 'w')
#    for pair in fdist.most_common(500):
#        s.write(str(pair[0])+'\n')
#    s.close()    
    
    #build Sparse Matrix Using Counte Vectorize
    count_vect = CountVectorizer(min_df = 1,stop_words = stopwords.words('english'))    
    xform = count_vect.fit_transform(big_feature_list)
    xform = preprocessing.scale(xform, with_mean=False)
    
    #break training and testing 
    Xtest = xform[500:]
    Ytest = label[500:]
    X = xform[0:500]
    Y = label[0:500]
    #initilize variables for cross validation
    V = 12
    M = len(Y)//V
    accLR = []
    accNB = []
    accSVM = []
    accBAGSVM = []
    accBAGLR = []
    accmajor = []
    #cross Validate
    for fold in range(V):
        first, last = fold*M, (fold+1)*M
        xtest , xtrain = X[first:last], vstack([X[:first],X[last:]])
        ytest , ytrain = Y[first:last], Y[:first]+Y[last:]
        #Machine learning Algorithm
        clfSVM = svm.LinearSVC(C =1).fit(xtrain, ytrain)
        clfLR = LogisticRegression(penalty = 'l2', C = 1).fit(xtrain, ytrain)
        clfNB = MultinomialNB(alpha = 0.01).fit(xtrain, ytrain)
        clfBAGSVM = BaggingClassifier(base_estimator = svm.SVC(kernel = 'linear', C = 1), n_estimators = 50, max_samples = .75,max_features = .75).fit(xtrain,ytrain)    
        clfBAGLR = BaggingClassifier(base_estimator = LogisticRegression(penalty = 'l2', C =1), n_estimators = 50, max_samples = .75,max_features = .75).fit(xtrain,ytrain)
        #predict on validation sample        
        yhatSVM=clfSVM.predict(xtest)
        yhatLR=clfLR.predict(xtest)
        yhatNB=clfNB.predict(xtest)
        yhatBAGSVM = clfBAGSVM.predict(xtest)
        yhatBAGLR = clfBAGLR.predict(xtest)
        ymajority = [1 if (yhatSVM[i]+yhatLR[i]+yhatNB[i]+yhatBAGSVM[i]+yhatBAGLR[i] > 0) else -1 for i in range(len(ytest))]  
        ymajority = np.asarray(ymajority)
        #Error Rate
        accmajor.append(np.mean(ymajority==ytest))
        accLR.append(np.mean(yhatLR==ytest))
        accSVM.append(np.mean(yhatSVM==ytest))
        accNB.append(np.mean(yhatNB==ytest))
        accBAGSVM.append(np.mean(yhatBAGSVM==ytest))
        accBAGLR.append(np.mean(yhatBAGLR==ytest))
        
    print()
    print('SVM ACCURACY Train:')
    print(accSVM)    
    print(sum(accSVM)/len(accSVM))
    print('LR ACCURACY Train:')
    print(accLR)
    print(sum(accLR)/len(accLR))
    print('NB ACCURACY Train:')
    print(accNB)
    print(sum(accNB)/len(accNB))
    print('BAG SVM ACCURACY Train:')
    print(accBAGSVM)
    print(sum(accBAGSVM)/len(accBAGSVM))
    print('BAG LR ACCURACY Train:')
    print(accBAGLR)
    print(sum(accBAGLR)/len(accBAGLR))
    print('Majority ACCURACY Train:')
    print(accmajor)
    print(sum(accmajor)/len(accmajor))
    print()
    
    #train on all data 
    clfSVM = svm.LinearSVC(C =1).fit(X, Y)
    clfLR = LogisticRegression(penalty = 'l2', C = 1).fit(X, Y)
    clfNB = MultinomialNB(alpha = 0.01).fit(X, Y)
    clfBAGSVM = BaggingClassifier(base_estimator = svm.SVC(kernel = 'linear', C = 1), n_estimators = 50, max_samples = .75,max_features = .75).fit(X, Y)   
    clfBAGLR = BaggingClassifier(base_estimator = LogisticRegression(penalty = 'l2', C =1), n_estimators = 50, max_samples = .75,max_features = .75).fit(X, Y)
    #predict on all data    
    yhatSVM=clfSVM.predict(Xtest)
    yhatLR=clfLR.predict(Xtest)
    yhatNB=clfNB.predict(Xtest)
    yhatBAGSVM = clfBAGSVM.predict(Xtest)
    yhatBAGLR = clfBAGLR.predict(Xtest)
    ymajority = [1 if (yhatSVM[i]+yhatLR[i]+yhatNB[i]+yhatBAGSVM[i]+yhatBAGLR[i] > 0) else -1 for i in range(len(Ytest))]  
    ymajority = np.asarray(ymajority)
    
    print('Training on all data')
    print('SVM ACCURACY Test:')
    print(np.mean(yhatSVM==Ytest))
    print('LR ACCURACY Test:')
    print(np.mean(yhatLR==Ytest))
    print('NB ACCURACY Test:')
    print(np.mean(yhatNB==Ytest))
    print('BAG SVM ACCURACY Test:')
    print(np.mean(yhatBAGSVM==Ytest))
    print('BAG LR ACCURACY Test:')
    print(np.mean(yhatBAGLR==Ytest))
    print('Majority ACCURACY Test:')
    print(np.mean(ymajority==Ytest))
    print()