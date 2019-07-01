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
Изначально, нужно подключить необходимые библиотеки и функции, которые будут задействованны в написании программы cancer_gnb.py. Библиотека Pandas нам потребуется для импорта данных раковых больных из "wdbc.data". Из бибилиотеки sklearn подключаем модуль GaussianNB, который является ключевым для рассчета данных. Далее, мы подключаем написанную функцию cross_validation_plotter, которая необходима для построения графиков. cross_val_score используется для оценки модели прогнозирования. Далее мы выгружаем данные и структуризируем их для дальнейшей работы. Мы назвали экземпляр нашего алгоритма оценки clf, так как он является классификатором. Теперь он должен быть применен к модели, т.е. он должен обучится на модели. Это осуществляется путем прогона нашей обучающей выборки через метод fit. Следом идет вывод полученных результатов.

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
Сперва, как и в прошлом примере, подключаем необходимые библиотеки. Далее нам потребуется  Pandas для импорта данных раковых больных из "wdbc.data". Из бибилиотеки sklearn подключаем модуль LinearDiscriminantAnalysisB для рассчета данных. Далее подключаем написанную функцию cross_validation_plotter, которая необходима для построения графиков. cross_val_score используется для оценки модели прогнозирования. Далее мы выгружаем данные и структуризируем их для дальнейшей работы. Мы назвали экземпляр нашего алгоритма оценки clf, так как он является классификатором. Теперь он должен быть применен к модели, т.е. он должен обучится на модели. Это осуществляется путем прогона нашей обучающей выборки через метод fit. Следом идет вывод полученных результатов.

# Время работы каждого из кодов
|     Программа    | Процент ошибок для тестовых данных | Процент ошибок для обучающей выборки | Время работы в секундах |
|:----------------:|:----------------------------------:|:------------------------------------:| :------------------------------------:|
|  cancer_gnb.py | 0,93 | 0,91 | 0.08 |
|   cancer_linear.py   | 0,94 | 0,93| 0.07 |
| diabets_gnb.py   | 1,0 | 1,0 | 0.05 |
| diabets_linear.py | 0,7 | 0,7 | 0.06 |
| digits_gnb.py      | 0,84 | 0,86 | 0.29 |
| digits_linear.py      | 0,95 | 0,97 | 0.58 |
| iris_gnb.py         | 0,95 | 0,95 | 0.04 |
| iris_linear.py       | 0,96 | 0,98 | 0.02 |
| titanic_gnb.py     | 0,71 | 0,75 | 0.05 |
| titanic_linear.py         | 0,74 | 0,78 | 0.05 |

