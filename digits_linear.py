from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from plotter import plot_cross_validation
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')
#------------------
digits = datasets.load_digits()
X = digits.data
Y = digits.target

clf=LinearDiscriminantAnalysis()
clf.fit(X,Y)

n = 1
print(clf.predict(X[1:10]))
print((X[1:3]))

print(cross_val_score(clf, X, Y, cv=10))

print("Сравнение показателей ...")
plot_cross_validation(X=X, y=Y, clf=clf, title="Linear Discriminant Analysis")