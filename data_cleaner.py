import pandas as pd
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


class DataCleaner:


    # This function will be called automatically when you intantie the class
    def __init__(self):
        self.cm1=pd.read_csv("datasets/cm1.csv")
        self.kc1=pd.read_csv("datasets/kc1.csv")




    def getCm1(self):
        return self.cm1





    def getKc1(self):
        return self.kc1





    def visualize(self, set_type):
            if set_type=='kc1':
                self.kc1['defects'].value_counts().sort_index().plot.bar()
            else:
                self.cm1['defects'].value_counts().sort_index().plot.bar()
            plt.show()




    def getCols(self,set_type):
            if set_type=='kc1':
                return self.kc1.columns
            else:
                return self.cm1.columns


