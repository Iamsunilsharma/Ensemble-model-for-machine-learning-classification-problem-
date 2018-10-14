import numpy as np
from sklearn import preprocessing, cross_validation,neighbors,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd

df=pd.read_csv('data.csv')
df.drop(['s.no'],1,inplace=True)
df.drop(['quality'],1,inplace=True)
X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
bg.fit(X_train,y_train)
accuracy1=bg.score(X_test,y_test)
print(accuracy1)

adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 10, learning_rate = 1)
adb.fit(X_train,y_train)
accuracy = adb.score(X_test,y_test)
print(accuracy)

gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
accuracy3 = gbc.score(X_test,y_test)
print(accuracy3)



example_measure=np.array([[1,76,71,71,71,60,47,25.141784,1.054384,9.874633,0.09978,0.023386,0,0,0,0.560959,0.109134,0]])
#print(len(example_measure))
example_measure=example_measure.reshape(len(example_measure),-1)
prediction=gbc.predict(example_measure)
print(prediction)
