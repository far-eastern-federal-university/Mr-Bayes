# Mr-Bayes


Выполнили студентки группы Б8320:
Волосевич Валерия,
Рудь Мария,
Шавырова Екатерина.


По результатам выполненных работ мы имеем 10 рабочих программ:
- cancer_gnb.py (Наивный байесовский классификатор для набора данных по раковым больным)
- cancer_linear.py (Линейный дискриминант Фишера для набора данных по раковым больным)
- diabets_gnb.py (Наивный байесовский классификатор для наборы данных по диабету)
- diabets_linear.py (Линейный дискриминант Фишера для набора данных по диабету)
- digits_gnb.py (Наивный байесовский классификатор для набора чисел)
- degits_linear.py (Линейный дискриминант Фишера для набора чисел)
- iris_gnb.py (Наивный байесовский классификатор для ирисов)
- iris_linear.py (Линейный дискриминант Фишера для ирисов)
- titanic_gnb.py (Наивный байесовский классификатор для данных по пассажирам Титаника)
- titanic_linear.py (Линейный дискриминант Фишера для данных по пассажирам Титаника)

# Описание кода для метода наивного байесовского классификатора
Для объяснения принципа работы метода логистической регерессии мы привели пример на программе cancer_gnb.py. В других программах, использующих данный метод, меняется только блок выгрузки данных.

### Описание кода cancer_gnb.py
```sh
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
```
Изначально, нужно подключить необходимые библиотеки и функции, которые будут задействованны в написании программы cancer_gnb.py. Библиотека Pandas нам потребуется для импорта данных раковых больных из "wdbc.data". 

# Описание кода для метода линейного дискриминанта Фишера
Также, как и в предыдущем случае, для объяснения принципа работы метода опорных векторов мы привели пример на программе cancer_linear.py. В других программах, использующих данный метод, меняется, только блок выгрузки данных.

### Описание кода cancer_linear.py
```sh
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from plotter import plot_cross_validation
import warnings
warnings.filterwarnings('ignore')
#------------------
data = pd.read_csv("wdbc.data", sep = ",")
data = data.iloc[:,0:12]
X = data.iloc[:,2:12]
Y = data.iloc[:,1]

clf=LinearDiscriminantAnalysis()
clf.fit(X,Y)
print(np.array(clf.predict([X.loc[1]])))

print("Where is malignant cancer?")
n = 1
print(clf.predict(X[14:23]))
print((X[14:23]))

print(np.array(clf.predict([X.loc[1]])))
print(cross_val_score(clf, X, Y, cv=10))


print("Сравнение показателей ...")
plot_cross_validation(X=X, y=Y, clf=clf, title="Linear Discriminant Analysis")
```


# Время работы каждого из кодов
|     Программа    | Процент ошибок для тестовых данных | Процент ошибок для обучающей выборки |
|:----------------:|:----------------------------------:|:------------------------------------:|
|  cancer_gnb.py |  |  |
|   cancer_linear.py   |  |  |
| diabets_gnb.py   |  |  |
| diabets_linear.py |  |  |
| digits_gnb.py      |  |  |
| digits_linear.py      |  |  |
| iris_gnb.py         |  |  |
| iris_linear.py       |  |  |
| titanic_gnb.py     |  |  |
| titanic_linear.py         |  |  |

