from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from plotter import plot_cross_validation
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')
#------------------
data = datasets.load_iris()
X=data.data
Y=data.target

clf=LinearDiscriminantAnalysis()

clf.fit(X, Y)
print(clf.predict(X[:5]))
print((X[:5]))

print(cross_val_score(clf, X, Y, cv=10))

print("Сравнение показателей ...")
plot_cross_validation(X=X, y=Y, clf=clf, title="Linear Discriminant Analysis")