"""
Using SVM for Email Spam Classification
"""

"""
Author Details:

R Anirudh     (1MS17IS084)

Rohit P N     (1MS17IS094)

Snehil Tiwari (1MS17IS153)

Institute: Ramaiah Institite of Technology, Bangalore

Date of submission: 16 May 2020
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

seed = 0


data = pd.read_csv('./data/spambase.data', header=None)
data = data.rename({57: "label"}, axis=1)

print("Close the diagram to continue")
data['label'].value_counts().plot(kind='pie', cmap='RdPu')
plt.show()


def report(y_true, y_pred, C,  title):

    target_labels = ['class_0', 'class_1']

    classificationReport = classification_report(
        y_true, y_pred, target_names=target_labels)
    confusionMatrix = confusion_matrix(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)

    print('\n-------------------------------------------',
          title, '-------------------------------------------')

    print('\n\n------------------------ OPTIMUM C VALUE (REGULARIZATION TERM) -------------------------')
    print(C)

    print('\n\n------------------------ TEST ACCURACY -------------------------')
    print(oa)

    print("\n\n---------------------- CLASSIFICATION REPORT ----------------------")
    print(classificationReport)

    print("\n\n------------------------ CONFUSION MATRIX -------------------------")
    print(confusionMatrix)
    sns.heatmap(confusionMatrix, annot=True, cmap="RdPu", xticklabels=[
                'Not Spam (Actual)', 'Spam (Actual)'], yticklabels=['Not Spam (Predicted)', 'Spam (Predicted)'])


X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

print("Training...")
C_list = []
accuracies = []
for C in np.arange(1, 200000, 20000):

    classifier = SVC(C=C, kernel='rbf', random_state=seed)
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)

    C_list.append(C)
    accuracies.append(acc)

plt.plot(C_list, accuracies)
plt.xlabel(' Regularization Parameter C')
plt.ylabel('Test Accuracy')
plt.show()

C = C_list[np.argmax(accuracies)]

classifier = SVC(C=C, kernel='rbf', random_state=seed)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

report(y_test, y_pred, C, title='RBF KERNEL WITH DATA IMBALANCE')

C_list = []
accuracies = []
for C in np.arange(1, 100, 10):

    classifier = SVC(C=C, kernel='linear', max_iter=10000, random_state=seed)
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)

    C_list.append(C)
    accuracies.append(acc)

plt.plot(C_list, accuracies)
plt.xlabel(' Regularization Parameter C')
plt.ylabel('Test Accuracy')
plt.show()

C = C_list[np.argmax(accuracies)]

classifier = SVC(C=C, kernel='linear', max_iter=10000, random_state=seed)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

report(y_test, y_pred, C, title='LINEAR KERNEL WITH DATA IMBALANCE')

C_list = []
accuracies = []
for C in np.arange(1, 10000, 1000):

    classifier = SVC(C=C, kernel='poly', degree=2,
                     max_iter=100000, random_state=seed)
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)

    C_list.append(C)
    accuracies.append(acc)

plt.plot(C_list, accuracies)
plt.xlabel(' Regularization Parameter C')
plt.ylabel('Test Accuracy')
plt.show()

C = C_list[np.argmax(accuracies)]

classifier = SVC(C=C, kernel='poly', degree=2,
                 max_iter=100000, random_state=seed)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

report(y_test, y_pred, C, title='QUADRATIC KERNEL WITH DATA IMBALANCE')

data_1 = data[data['label'] == 1]

data_0 = data[data['label'] == 0].sample(n=len(data_1))

data = pd.concat([data_0, data_1]).reset_index()

data['label'].value_counts().plot(kind='pie', cmap='RdPu')
plt.show()

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed)

C_list = []
accuracies = []
for C in np.arange(1, 200000, 20000):

    classifier = SVC(C=C, kernel='rbf', random_state=seed)
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)

    C_list.append(C)
    accuracies.append(acc)

plt.plot(C_list, accuracies)
plt.xlabel(' Regularization Parameter C')
plt.ylabel('Test Accuracy')
plt.show()

C = C_list[np.argmax(accuracies)]

classifier = SVC(C=C, kernel='rbf', random_state=seed)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

report(y_test, y_pred, C, title='RBF KERNEL WITH BALANCED DATA')

C_list = []
accuracies = []
for C in np.arange(1, 10, 1):

    classifier = SVC(C=C, kernel='linear', random_state=seed)
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)

    C_list.append(C)
    accuracies.append(acc)

plt.plot(C_list, accuracies)
plt.xlabel(' Regularization Parameter C')
plt.ylabel('Test Accuracy')
plt.show()

C = C_list[np.argmax(accuracies)]

classifier = SVC(C=C, kernel='linear', random_state=seed)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

report(y_test, y_pred, C, title='LINEAR KERNEL WITH BALANCED DATA')

C_list = []
accuracies = []
for C in np.arange(1, 200000, 20000):

    classifier = SVC(C=C, kernel='poly', degree=2, random_state=seed)
    classifier.fit(X_train, y_train)
    acc = classifier.score(X_test, y_test)

    C_list.append(C)
    accuracies.append(acc)

plt.plot(C_list, accuracies)
plt.xlabel(' Regularization Parameter C')
plt.ylabel('Test Accuracy')
plt.show()

C = C_list[np.argmax(accuracies)]

classifier = SVC(C=C, kernel='poly', degree=2, random_state=seed)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

report(y_test, y_pred, C, title='QUADRATIC KERNEL WITH BALANCED DATA')
