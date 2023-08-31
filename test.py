from flask import Flask, render_template,request
import numpy as np


# Import neccessary library and the preprocessed dataset
import pandas as pd
# libaries for evaluating the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Libraries for building classification model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
try:
    df = pd.read_csv("preprocessed_dataset.csv")
    print("dataset successfully loaded")
except (FileNotFoundError, IOError):
    print("Wrong file or invalid file path")

predictors = ['num_of_prev_attempts', 'average_score', 'sum_click', 'studied_credits']
X = df[predictors]
# Assign the final_result column as target feature
y = df['final_result']

# Import the necessary module
from sklearn.model_selection import train_test_split

# Split data set into train and test sets X_train, X_test, y_train, y_test
target_test = train_test_split(X, y, test_size=0.30, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

svc_model = SVC(random_state=0, class_weight='balanced')

fr_model = RandomForestClassifier(n_estimators=1000, random_state=0)
# Train the algorithm on training data and predict using the testing data
pred = fr_model.fit(X_train, y_train).predict(X_test)

# Import Pickle module
import pickle
# Save model
filename = 'trained_model.sav'
pickle.dump(fr_model, open(filename, 'wb'))
load_lr_model = pickle.load(open(filename, 'rb'))
# Load model
load_lr_model = pickle.load(open('trained_model.sav', 'rb'))


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


app = Flask(__name__)
load_lr_model = pickle.load(open('trained_model.sav', 'rb'))

#dash app starts

from dash import Dash

# Create Dash app within Flask app
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Import layout and callbacks from dash_app.py
from dash_app import app as dash_app_instance
dash_app.layout = dash_app_instance.layout
#dash app ends


@app.route('/')
def index_page():

    return render_template('index.html')

@app.route('/home')
def goto_home():
    return render_template('index.html')

@app.route('/predictpage')
def predictionPage():
    return render_template('predictpage.html')

@app.route('/upload')
def uploadFile():
    return render_template('upload.html')

#Visualization
@app.route('/graph')
def showGraph():
    return render_template('graph.html')



#upload file
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    data = None
    columns = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the uploaded CSV file into a Pandas DataFrame
            data = pd.read_csv(file)
            columns = data.columns.tolist()

    return render_template('upload.html', data=data.values.tolist() if data is not None else None, columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Define a dictionary to map numeric predictions to labels
    prediction_labels = {
        0: 'pass',
        1: 'fail',
        2: 'distinction',
        3: 'withdrawn'
    }
    int_features = [int(x) for x in request.form.values()]
    #selected_imd_band = request.form['algo']
    final_features = [np.array(int_features)]
    prediction = load_lr_model.predict(final_features)
    output = round(prediction[0], 2)
    # Convert numeric prediction to label
    readable_prediction = prediction_labels[output]

    return render_template('predictpage.html', prediction_text=' The final performance is {}'.format(readable_prediction))

@app.route('/analysis')
def analyse():
    return render_template('analysis.html')

@app.route('/analyse', methods=['POST'])

def result():
    #analysis of the prediction model based on the respond obtained from the form
    analyse_text = request.form['algo']
    if analyse_text == 'Naïve Bayes':
        gnb_clf = GaussianNB()
        gnb_clf.fit(X_train, y_train)
        result1 = getScore(gnb_clf, X_train, y_train, X_test, y_test, train =True)
        result2 = getScore(gnb_clf, X_train, y_train, X_test, y_test, train=False)
        return render_template('analysis.html', result =result2)

    elif analyse_text == 'Logistic Regression':
        lr_clf = LogisticRegression(solver='liblinear')
        lr_clf.fit(X_train, y_train)
        # Train result
        lrResultTrain= getScore(lr_clf, X_train, y_train, X_test, y_test, train=True)
        # Test result
        lrResultTest =getScore(lr_clf, X_train, y_train, X_test, y_test, train=False)
        return render_template('analysis.html', result=lrResultTest)
    elif analyse_text == 'Support Vector Machine':
        svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
        svm_clf.fit(X_train, y_train)
        svmResultTrain = getScore(svm_clf, X_train, y_train, X_test, y_test, train=True)
        svmResultTest =getScore(svm_clf, X_train, y_train, X_test, y_test, train=False)
        return render_template('analysis.html', result=svmResultTest)
    elif analyse_text == 'Random Forest classifier':
        rf_clf = RandomForestClassifier(n_estimators=1000, random_state=0)
        rf_clf.fit(X_train, y_train)
        rfResultTrain= getScore(rf_clf, X_train, y_train, X_test, y_test, train=True)
        rfResultTest=getScore(rf_clf, X_train, y_train, X_test, y_test, train=False)
        return render_template('analysis.html', result=rfResultTest)
    elif analyse_text == 'K-Nearest Neighbour':
        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_train, y_train)
        #knnResultTrain = getScore(knn_clf, X_train, y_train, X_test, y_test, train=True)
        knnResultTest =getScore(knn_clf, X_train, y_train, X_test, y_test, train=False)
        return render_template('analysis.html', result=knnResultTest)

