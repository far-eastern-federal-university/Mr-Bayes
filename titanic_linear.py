import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from plotter import plot_cross_validation
import warnings
warnings.filterwarnings('ignore')
#------------------
data = pd.read_csv("titanic.csv", sep = ",")
data_notna = pd.DataFrame.dropna(data)
X = data_notna[["Pclass", "Fare", "Age", "Sex"]]
X.replace("male", 0, True, None, False)
X.replace("female", 1, True, None, False)

Y = data_notna["Survived"]
clf=LinearDiscriminantAnalysis()
clf.fit(X,Y)
print(np.array(clf.predict([X.loc[1]])))

print("Do they survived?")
n = 1
print(clf.predict(X[5:10]))
print((X[5:10]))

print(np.array(clf.predict([X.loc[1]])))
print(cross_val_score(clf, X, Y, cv=10))

print("Сравнение показателей ...")
plot_cross_validation(X=X, y=Y, clf=clf, title="Linear Discriminant Analysis")