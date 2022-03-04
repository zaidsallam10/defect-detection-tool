
from hashlib import new
from unicodedata import name
# sklearn: sikit-learn/learn; its a python library dedicated to call and use machine learning algorithms
# sklearn: contains:
# 1. Data preprocessing functions
# 2. Data Splitting functions (70% for training 15% testing 15% validation)
# 3. Machine learning algorthims (DT: Decision Tree)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler





def checkColumnDataType(col):
    return  col == 'float64'





def scaleColumns(df, cols):
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    df[cols]=scaler.fit_transform(df[cols])
    return df




# desctipion; it provides you with the min,max,avg, median, Stander derivation
def printColDescrption(col):
    print(col.describe())







# this function will loop over columns and scale them
# it takes the 1. dataset / 2. columns / 3. exported file name
def scallingValues(df, columns, exported_file_name):



    # Before Scalling
    numeric_cols=[]
    for col in columns:
        print("-----------------------------")
        print("Before ;Column Name: ",col)
        print("Before; Column Description: ", checkColumnDataType(df[col].dtype) )
        print(printColDescrption(df[col]))
        if checkColumnDataType(df[col].dtype):
            numeric_cols.append(col)
        print("-----------------------------")
        

    # After Scalling
    new_df=scaleColumns(df, numeric_cols)
    for col in new_df.columns:
        print("After ;Column Name: ",col)
        print("After; Column Description: ", printColDescrption(new_df[col]) )
    
    # export new dataset
    new_df.to_csv(exported_file_name, index=False)


