import numpy as np
from sklearn import preprocessing, cross_validation,neighbors,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd

df=pd.read_csv('data.csv')
df.drop(['s.no'],1,inplace=True)
df.drop(['quality'],1,inplace=True)
df.drop(['ps'],1,inplace=True)
df.drop(['amfm'],1,inplace=True)
df.drop(['distance'],1,inplace=True)


#df.drop(['diameter'],1,inplace=True)


X=np.array(df.drop(['class'],1))
y=np.array(df['class'])
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.1)


#NN1 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(200), random_state=100,learning_rate='invscaling')


kfold = model_selection.KFold(n_splits=10)
model1 = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(140), random_state=110,learning_rate='invscaling')
results1 = model_selection.cross_val_score(model1, X, y, cv=kfold)
print(results1)
print(results1.mean()*100)

model2 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(140), random_state=110,learning_rate='invscaling')
results2 = model_selection.cross_val_score(model2, X, y, cv=kfold)
print(results2)
print(results2.mean()*100)

model3 = MLPClassifier(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(140), random_state=100,learning_rate='invscaling')
results3 = model_selection.cross_val_score(model3, X, y, cv=kfold)
print(results3)
print(results3.mean()*100)

model4 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(140), random_state=100,learning_rate='invscaling')
results4 = model_selection.cross_val_score(model4, X, y, cv=kfold)
print(results4)
print(results4.mean()*100)



'''


model.fit(X_train,y_train)
accuracy=model.score(X_test,y_test)
print(accuracy)
'''
