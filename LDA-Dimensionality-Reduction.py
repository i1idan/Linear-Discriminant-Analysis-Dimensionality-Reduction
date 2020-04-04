
#LDA-For-Dimensionality-Reduction
#70_30_tran_test_split

"""
***Parameters***

X=FeatureVector
y=Labels

"""

#Importing Libraries

import numpy as np
import pandas as pd

#Data preprocessing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Performing LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#Training
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=3, random_state=20)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Evalution

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))
