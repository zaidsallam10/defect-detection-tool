
import numpy as np
import pandas as pd

from data_cleaner import *


# importing the cleaner class
c =DataCleaner()
# calling the get dataset
print(c.getKc1())

#visualization of the defect frequency by dataset name
# c.visualize("cm1")

# printing the columns of the set by name
c.getCols("cm1")

