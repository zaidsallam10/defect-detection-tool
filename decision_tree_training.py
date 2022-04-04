# For algorithm training 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# For data splitting
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pickle
# from sklearn.cross_validation import train_test_split
import sklearn
# from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix


# non-scalled-dataset- path: datasets/kc1.csv
# scalled-dataset- path: scalled_kc1.csv


dataset= pd.read_csv("datasets/kc1.csv")


# dataset= pd.read_csv("datasets/kc1.csv")

# dataset= pd.read_csv("scalled_kc1_v2.csv")
# dataset= pd.read_csv("scalled_cm1_v2.csv")

# the column name of the target / label
y=dataset['defects']
# the columns of the features
X=dataset.drop(["defects"], axis=1)

# essential_complexity
# branchCount_of_the_flow_graph

# we split the data into testing and training
# training data used to train the algorithm (During training)
# testing data used to test and get the accuracy of the model (After training)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

# X_1, X_test, Y_1, Y_test = train_test_split(X, y, test_size=0.0, random_state=0)
# X_train,X_cv,Y_train,Y_cv = train_test_split(X_1,Y_1,test_size=0.0,random_state=0)



# scaler = StandardScaler()
# # Fit only to the training d
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_cv = scaler.transform(X_cv)
# X_test = scaler.transform(X_test)
# X_train =  pd.DataFrame(data=normalize(X_train))
# X_cv = pd.DataFrame(data=normalize(X_cv))
# X_test = pd.DataFrame(data=normalize(X_test))



# For training
clf = DecisionTreeClassifier()
clf.fit(X,y)

filename = 'decision_tree_clf_kc1_2.sav'
pickle.dump(clf, open(filename, 'wb'))

# # # For testing
# # score=clf.score(X_test,y_test)
# # print(score)

# y_score = clf.fit(X_train,Y_train)
# p=clf.predict(X_test)
# acc = accuracy_score(Y_test,p)
# print(acc)








# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# mlp = MLPClassifier(hidden_layer_sizes=(7,7,7),max_iter=400,activation='relu')
# y_score = mlp.fit(X,y)
# p=mlp.predict(X_test)
# acc = accuracy_score(Y_test,p)
# con_matrix=confusion_matrix(Y_test,p)
# roc = roc_curve(Y_test,p)
# fr, tr, thresholds = metrics.roc_curve(Y_test, p)
# #rms = (Y_test.T - p) ** 2
# for i in range(len(Y_test)):
#     fpr[i], tpr[i], _ = roc_curve(Y_test[:], p)
#     roc_auc[i] = auc(fpr[i], tpr[i])
# print('Accuracy of network over test data is: ',acc*100)
# print('<----------------------------------------------------------------------------------->')
# print('Confusion matrix: ')
# print(con_matrix)
# print('<----------------------------------------------------------------------------------->')

# print('True positive rate: ',tr[2])
# print('False positive rate: ',fr[2])
# print('<----------------------------------------------------------------------------------->')



# filename = 'decision_tree_mlp2.sav'
# pickle.dump(mlp, open(filename, 'wb'))




# y_pred = clf.predict(X_test)


# Calculate the confusion matrix
#
# conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

# kc1= 125



# conf_matrix=np.array([np.array([64,1]),np.array([3,57])])
# #   [[200,30],[10,20]]

# print("conf_matrix=",conf_matrix)
# print("conf_matrix shape=",conf_matrix.shape)
# #
# # Print the confusion matrix using Matplotlib
# #
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()