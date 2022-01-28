
from hashlib import new
from unicodedata import name
from sklearn.preprocessing import StandardScaler



def checkColumnDataType(col):
    return  col == 'float64'



def scaleColumns(df, cols):
    scaler = StandardScaler()
    df[cols]=scaler.fit_transform(df[cols])
    return df


def printColDescrption(col):
    print(col.describe())



# this function will loop over columns and scale them
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


