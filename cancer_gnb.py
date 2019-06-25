import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from plotter import plot_cross_validation
import warnings
warnings.filterwarnings('ignore')
#------------------
data = pd.read_csv("wdbc.data", sep = ",")
data = data.iloc[:,0:12]
X = data.iloc[:,2:12]
Y = data.iloc[:,1]

clf = GaussianNB()
partial = clf.partial_fit
print(partial)
clf.fit(X, Y)

print("Where is malignant cancer?")
n = 1
print(clf.predict(X[14:23]))
print((X[14:23]))

print(clf.partial_fit)
print(cross_val_score(clf, X, Y, cv=10))
print("Сравнение показателей ...")
plot_cross_validation(X=X, y=Y, clf=clf, title="Gaussian Naive Bayes")