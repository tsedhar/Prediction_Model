#importing required libraries

import pandas as pd

#train and test function
from sklearn.model_selection import train_test_split

#import ML model algorithmn
from sklearn.svm import SVC
#libaries for evaluating the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    df = pd.read_csv("preprocessed_dataset.csv")
    print("dataset successfully loaded")
except (FileNotFoundError, IOError):
    print("Wrong file or invalid file path")

#Selection of independent variables
predictors = ['num_of_prev_attempts', 'average_score', 'sum_click', 'studied_credits']
X = df[predictors]
#Assign the final_result column as target feature or dependent variable
y = df['final_result']

#Split data set into train and test sets X_train, X_test, y_train, y_test
target_test = train_test_split(X,y, test_size = 0.30, random_state = 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = 20)

#building model
svc_model = SVC(random_state=0)
#Train the algorithm on training data and predict using the testing data
pred = svc_model.fit(X_train, y_train).predict(X_test)

#Import Pickle module
import pickle
#Save model

filename='trained_model.sav'
pickle.dump(svc_model, open(filename, 'wb'))
load_lr_model =pickle.load(open(filename, 'rb'))
print(load_lr_model.predict([[pred]]))

def getScore(classification, X_train, Y_train, X_test, Y_test, train=True):
    if train:
        pred = classification.predict(X_train)
        clf_report = pd.DataFrame(classification_report(Y_train, pred, output_dict=True))
        print("=========================   Train Result  =======================\n")
        print(f"Accuracy Score: {accuracy_score(Y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(Y_train, pred)}\n")

    elif train == False:
        pred = classification.predict(X_test)
        clf_report = pd.DataFrame(classification_report(Y_test, pred, output_dict=True))
        print("======================= Test Result =========================\n")
        print(f"Accuracy Score: {accuracy_score(Y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(Y_test, pred)}\n")



