# Steps:
    # Load Data.
    # Define Keras Model.
    # Compile Keras Model.
    # Fit Keras Model.
    # Evaluate Keras Model.
    # Tie It All Together.
    # Make Predictions
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset= pd.read_csv("datasets/kc1.csv")
y=dataset['defects']
X=dataset.drop("defects", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0)

# print(X_train.shape)
# print(y_train.shape)

model = Sequential()
# Input layer (input_dim=> number of features (X))
model.add(Dense(12, input_dim=21, activation='relu'))
# Hidden layer (Number of hiddern layers depend on the problem we're trying to solve)
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
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
model.fit(X, y, epochs=1000, batch_size=4)


# Testing and get the accuracy
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))