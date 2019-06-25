import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from plotter import plot_cross_validation
import warnings
warnings.filterwarnings('ignore')
#------------------
data = pd.read_csv("diabetes.csv", sep = ",")
data = data.iloc[:,0:12]
X = data.iloc[:,2:12]
Y = data.iloc[:,8]

clf=LinearDiscriminantAnalysis()
clf.fit(X,Y)
print(np.array(clf.predict([X.loc[1]])))

print("have diabetes or not?")
n = 1
print(clf.predict(X[14:23]))
print((X[14:23]))

print(np.array(clf.predict([X.loc[1]])))
print(cross_val_score(clf, X, Y, cv=10))

print("Сравнение показателей ...")
plot_cross_validation(X=X, y=Y, clf=clf, title="Linear Discriminant Analysis")