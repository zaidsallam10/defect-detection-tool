
import numpy as np
import pandas as pd

from data_cleaner import *
from data_preprocessing import *


# importing the cleaner class
c =DataCleaner()
# calling the get dataset
# print(c.getKc1())
# print(c.getCm1())

# print(c.getCols('mc1'))

#visualization of the defect frequency by dataset name
# c.visualize("kc1")

# printing the columns of the set by name
# c.getCols("cm1")


# this function will scale the numerical columns in the give dataset, and take it takes the exported new columns name
scallingValues(c.getKc1(), c.getCols('kc1'), 'scalled_kc1.csv')

scallingValues(c.getCm1(),c.getCols('mc1'), 'scalled_mc1.csv')
