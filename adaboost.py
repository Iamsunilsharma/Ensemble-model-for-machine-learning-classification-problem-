import numpy as np
from sklearn import preprocessing, cross_validation,neighbors,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor






df=pd.read_csv('data.csv')
df.drop(['s.no'],1,inplace=True)
df.drop(['quality'],1,inplace=True)
X=np.array(df.drop(['class'],1))
df.drop(['ps'],1,inplace=True)
df.drop(['amfm'],1,inplace=True)
df.drop(['distance'],1,inplace=True)
y=np.array(df['class'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)



DTC = RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=20, random_state=10)
DTC.fit(X_train,y_train)
accuracy=DTC.score(X_test,y_test)
print(accuracy)

clf7=AdaBoostClassifier(RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=20, random_state=10), n_estimators=100, learning_rate=1.0, random_state=10)



clf7.fit(X_train,y_train)
accuracy7=clf7.score(X_test,y_test)
print(accuracy7)
