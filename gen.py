# Простой вариант http://www.pymix.org/pymix/index.php?n=PyMix.Tutorial#quickstart

# Чуть более полезный
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Fun
def genpoints(listofdistr, numobj, cump):
    lst = []
    for i in range(numobj):
        j = np.argmax(np.array(cump) > np.random.uniform())
        r = eval('np.random.' + listofdistr[j][0] + '(' + str(listofdistr[j][1]) + ',' + str(listofdistr[j][2]) + ')')
        lst.append(r)
    return lst

# Блок определения входных данных
numobj = 1000
listofdistr = [['normal', 0, 1], ['uniform', 3, 5]]

# Фиксируем генератор случайных чисел
#np.random.seed(123)

# Генерация плотности распределения для списка распределений
w = np.random.uniform(size=len(listofdistr))
w = w/sum(w)

# Построение дискретной функции распределения
cump = np.cumsum(w)

# Проверка работы функции

lst = genpoints(listofdistr, numobj, w)
print(np.random.uniform(), np.random.uniform())

plt.hist(lst, bins='auto')

# Обратная задача: как получить значение функции плотности от x? (1d)

listofdistr = [['norm', 0, 1], ['uniform', 3, 5]]

def mix_distr(listofdistr, w, x):
    lst_val_distr = []
    for dstr in listofdistr:
        val = eval("scipy.stats." + dstr[0] + "(" + "x," + str(dstr[1]) + "," + str(dstr[2]) + ")")
        lst_val_distr.append(val)
    return np.dot(lst_val_distr, w)

mix_distr(listofdistr, w, 3)