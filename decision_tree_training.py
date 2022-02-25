# For algorithm training 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# For data splitting
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# non-scalled-dataset- path: datasets/kc1.csv
# scalled-dataset- path: scalled_kc1.csv

# dataset= pd.read_csv("scalled_kc1.csv")
# dataset= pd.read_csv("scalled_mc1.csv")
dataset= pd.read_csv("datasets/cm1.csv")
# dataset= pd.read_csv("datasets/kc1.csv")

# the column name of the target / label
y=dataset['defects']
# the columns of the features
X=dataset.drop("defects", axis=1)

# we split the data into testing and training
# training data used to train the algorithm (During training)
# testing data used to test and get the accuracy of the model (After training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# For training
clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
clf.fit(X_train, y_train)

# For testing
score=clf.score(X_test,y_test)
print(score)

y_pred = clf.predict(X_test)


# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()