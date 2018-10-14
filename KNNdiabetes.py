import numpy as np
from sklearn import preprocessing, cross_validation,neighbors,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
import pandas as pd

df=pd.read_csv('data.csv')
df.drop(['s.no'],1,inplace=True)
df.drop(['quality'],1,inplace=True)
X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.3)
clf=RandomForestClassifier(max_depth=2, random_state=0)
#clf=svm.SVC(kernel='linear')
#clf=tree.DecisionTreeClassifier()
print('hey')
clf.fit(X_train,y_train)
print('hello')
accuracy=clf.score(X_test,y_test)
print(accuracy)

example_measure=np.array([[1,76,71,71,71,60,47,25.141784,1.054384,9.874633,0.09978,0.023386,0,0,0,0.560959,0.109134,0]])
#print(len(example_measure))
example_measure=example_measure.reshape(len(example_measure),-1)
prediction=clf.predict(example_measure)
print(prediction)
