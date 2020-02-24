#importing the necessery libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)


#Cleaning the data
import re
import nltk
from nltk.corpus import stopwords
corpus = []
from nltk.stem.porter import PorterStemmer
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.metrics import confusion_matrix

##Implementing Naive bayes##

# Fitting Naive bayes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_naive = classifier.predict(X_test)

# Making the Confusion Matrix
cm_naive = confusion_matrix(y_test, y_pred_naive)

#Calculating accuracy
ar_naiveBayes = (cm_naive[0][0]+cm_naive[1][1])/200

##Implementing Decision Tree##

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_DT = classifier.predict(X_test)

# Making the Confusion Matrix
cm_DT = confusion_matrix(y_test, y_pred_DT)

#Calculating accuracy
ar_DT = (cm_DT[0][0]+cm_DT[1][1])/200

##Implementing Random Forest Tree##

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred_RFT = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RFT = confusion_matrix(y_test, y_pred_RFT)

#Calculating accuracy
ar_RFT = (cm_RFT[0][0]+cm_RFT[1][1])/200