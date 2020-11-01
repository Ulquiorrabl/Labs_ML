from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
from sklearn.linear_model import LogisticRegression


def TL():
    allurls = 'data.csv'  # path to our all urls file
    allurlscsv = pd.read_csv(allurls, ',', error_bad_lines=False)  # reading file
    allurlsdata = pd.DataFrame(allurlscsv)  # converting to a dataframe
    allurlsdata = np.array(allurlsdata)  # converting it into an array
    random.shuffle(allurlsdata)  # shuffling

    y = [d[1] for d in allurlsdata]  # all labels
    corpus = [d[0] for d in allurlsdata]  # all urls corresponding to a label (either good or bad)
    # tomized tokenizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)  # get the X vector
    # print(X[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)  # split into training and testing set 80/20 ratio

    lgs = LogisticRegression()  # using logistic regression
    lgs.fit(X_train, y_train)
    print(lgs.score(X_test, y_test))  # pring the score. It comes out to be 98%
    return vectorizer, lgs


vectorizer, lgs = TL()
# checking some random URLs. The results come out to be expected. The first two are okay and the last four are malicious/phishing/bad

X_predict = ['wikipedia.com', 'google.com/search=faizanahad', 'pakistanifacebookforever.com/getpassword.php/',
             'www.radsport-voggel.de/wp-admin/includes/log.exe', 'ahrenhei.without-transfer.ru/nethost.exe',
             'www.itidea.it/centroesteticosothys/img/_notes/gum.exe']

X_predict = vectorizer.transform(X_predict)
print(X_predict[1])

y_Predict = lgs.predict(X_predict)

print(y_Predict)  # printing predicted values
