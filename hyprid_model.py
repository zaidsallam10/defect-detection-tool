
# This file contains pipeline of two algorthims (DT-NN)


# For algorithm training 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# For data splitting
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Hyprid:

    def __init__(self,dataset):
        print("Welcome to hyprid DT-NN model")
        self.dataset= pd.read_csv(dataset)
        self.THRESHOLD=0
        self.selected_features=[]



    def decisionTree(self):
        # the column name of the target / label
        y=self.dataset['defects']
        # the columns of the features
        X=self.dataset.drop("defects", axis=1)
        # we split the data into testing and training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
        # For training
        clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
        clf.fit(X_train, y_train)
        # get importance
        importance = clf.feature_importances_
        # summarize feature importance
        for name, importance in zip(X.columns, importance):
            if(importance>self.THRESHOLD):
                # print(name, importance)
                self.selected_features.append(name)

        self.neuralNetwork()



    def neuralNetwork(self):
        print("Hola to neuralNetwork")
        X=self.dataset.drop("defects", axis=1)
        y=self.dataset['defects']
        print("This is the filtered X after selection the importance from the decision tree")
        filtered_X=X[self.selected_features]

        # Training the neural network based on importance from decision tree
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0)

        # print(X_train.shape)
        # print(y_train.shape)

        model = Sequential()
        # Input layer (input_dim=> number of features (X))
        model.add(Dense(256, input_dim=21, activation='relu'))
        # Hidden layer (Number of hiddern layers depend on the problem we're trying to solve)
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        # Output layer
        # sigmoid: is a amathatical function used to give the final output (mainly used for binary classification 0/1 True/False)
        model.add(Dense(1, activation='sigmoid'))


        # loss used to calculate the difference between real and predicted value (error= real - predicated)
        # loss used to calculate the difference between real and predicted value (binary_crossentropy)
        # optimizer: used to redefine and recalcualte the wights of the algirthim to give better result (adam)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Training 
        # epochs: loop over the training data to learn.
        # over fitting: The model can't learn anymore.
        model.fit(X_train, y_train, epochs=300, batch_size=4)


        # Testing and get the accuracy
        _, accuracy = model.evaluate(X_test, y_test)
        print('Accuracy of neural netwrok: %.2f' % (accuracy))

        model.save('hyprid_model.h5')  # creates a HDF5 file 'my_model.h5'





# model=Hyprid("scalled_kc1_v2.csv")
model=Hyprid("datasets/cm1.csv")
model.decisionTree()

# cm1=> 0.94
# scalled_cm1=> 0.96


# kc1=>  0.83
# scalled_kc1=> 0.87
