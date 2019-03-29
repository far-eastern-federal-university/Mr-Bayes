# Coming soon...

import pandas as pd
import numpy as np

numobj = 10
listofdistr = [['normal', 0, 1], ['uniform', 0, 1]]

print(eval('np.random.' + listofdistr[0][0] + '(' + str(listofdistr[0][1]) + ',' + str(listofdistr[0][2]) + ')'))
