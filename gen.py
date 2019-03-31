# Простой вариант http://www.pymix.org/pymix/index.php?n=PyMix.Tutorial#quickstart

# Чуть более полезный
import pandas as pd
import numpy as np

# Fun
def genpoints(listofdistr, numobj, cump):
    lst = []
    for i in range(numobj):
        j = np.argmax(np.array(cump) > np.random.uniform())
        r = eval('np.random.' + listofdistr[j-1][0] + '(' + str(listofdistr[j-1][1]) + ',' + str(listofdistr[j-1][2]) + ')')
        lst.append(r)
    return lst

# Блок определения входных данных
numobj = 10
listofdistr = [['normal', 0, 1], ['uniform', 0, 1]]

# Фиксируем генератор случайных чисел
#np.random.seed(123)

# Генерация плотности распределения для списка распределений
p = np.random.uniform(size=len(listofdistr))
p = p/sum(p)

# Построение дискретной функции распределения
cump = np.cumsum(p)

lst = genpoints(listofdistr, numobj, p)
print(lst)