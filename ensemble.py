import numpy as np
from sklearn import preprocessing, cross_validation,neighbors,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
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

BG = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 20)
RF = RandomForestClassifier(max_depth=2, random_state=0)
NN = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(130,), random_state=1)
SV = svm.SVC(kernel='linear')
ADA = AdaBoostClassifier()
KNN = KNeighborsClassifier()
DTC = tree.DecisionTreeClassifier()
MNB = MultinomialNB()
GBC = GradientBoostingClassifier()

evs = VotingClassifier(estimators=[('BG',BG),('RF',RF),('NN',NN),('SV',SV),('ADA',ADA),('KNN',KNN),('DTC',DTC),('MNB',MNB),('GBC',GBC)],voting='hard')
evs.fit(X_train,y_train)
accuracy=evs.score(X_test,y_test)
print(accuracy)


