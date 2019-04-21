#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Iris
#Импортим нужные библиотеки:
import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics 
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
# Загружаем набор данных:
iris = datasets.load_iris()
# Смотрим на названия переменных
print (iris.feature_names)
# Смотрим на данные, выводим 10 первых строк: 
print (iris.data[:10])
# Смотрим на целевую переменную:
print (iris.target_names)
print (iris.target)

iris_frame = DataFrame(iris.data)
# Делаем имена колонок такие же, как имена переменных:
iris_frame.columns = iris.feature_names
# Добавляем столбец с целевой переменной:  
iris_frame['target'] = iris.target
# Для наглядности добавляем столбец с сортами:  
iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x])
# Смотрим, что получилось:
print(iris_frame)

clf = GaussianNB()
partial = clf.partial_fit
print(partial)

data = load_iris()
X=data.data
Y=data.target
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict(X[:25]))
print((X[:25]))

print(clf.partial_fit)
print(cross_val_score(clf, X, Y, cv=10))

plt.figure()
plt.title("b")
plt.xlabel("Training examples")
plt.ylabel("Score")
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=241)
train_sizes=np.linspace(.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
        clf, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.legend(loc="best")

plt.show()


plt.plot(X, 'rs')
plt.plot(Y, 'bo')

plt.show()


# In[11]:


#Titanic
#Импортим нужные библиотеки:
import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics 
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

clf = GaussianNB()
partial = clf.partial_fit
print(partial)


data = pd.read_csv("titanic.csv", sep = ",")
print(data.describe()) # check if we've loaded right data

data_notna = pd.DataFrame.dropna(data)
print(data_notna.describe())

X = data_notna[["Pclass", "Fare", "Age", "Sex"]]
X.replace("male", 0, True, None, False)
X.replace("female", 1, True, None, False)
print(X[:5])

Y = data_notna["Survived"]

clf = GaussianNB()
clf.fit(X, Y)

print("Do they survived?")
n = 1
print(clf.predict(X[5:10]))
print((X[5:11]))

print(clf.partial_fit)
print(cross_val_score(clf, X, Y, cv=10))

plt.figure()
plt.title("b")
plt.xlabel("Training examples")
plt.ylabel("Score")
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=241)
train_sizes=np.linspace(.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
        clf, X, Y, cv=cv, n_jobs=4, train_sizes=train_sizes)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.legend(loc="best")

plt.show()


plt.plot(X, 'rs')
plt.plot(Y, 'bo')

plt.show()


# In[ ]:




