# dash_app.py
import dash
from dash import dcc
from dash import html,dash_table
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Libraries for building classification model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

app = dash.Dash(__name__)

# Load data from CSV
mainFile = pd.read_csv('preprocessed_dataset.csv')
stdInfo_df = pd.read_csv('studentInfo.csv')

# Separate features and target variable
X = mainFile.drop('final_result', axis=1)
y = mainFile['final_result']

# Train a RandomForestClassifier to get feature importances
model = RandomForestClassifier(n_estimators=1000, random_state=0)
model.fit(X, y)

ranked_features = pd.Series(model.feature_importances_, index=X.columns)
ranked_features = ranked_features.nlargest(9)  # Select top 9 features

# Convert Matplotlib visualization to Plotly
fig = px.bar(
    x=ranked_features.values,
    y=ranked_features.index,
    orientation='h',
    labels={'x': 'Feature Importance', 'y': 'Feature Names'},
    title='Random Forest Feature Importance'
)
#gender distribution

# Count the occurrences of each gender category
gender_counts = stdInfo_df['gender'].value_counts()

# Create a pie chart using Matplotlib
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')

# Convert Matplotlib plot to Plotly figure
pie_fig = go.Figure(go.Pie(labels=gender_counts.index, values=gender_counts, hole=0.3))

#Final result distribution
# Count the occurrences of each final result category
final_result_counts = stdInfo_df['final_result'].value_counts()

# Create a pie chart using Plotly
result_graph = go.Figure([go.Bar(x=final_result_counts.index, y=final_result_counts)])

#classification report
predictors = ['num_of_prev_attempts', 'average_score', 'sum_click', 'studied_credits']
X = mainFile[predictors]
# Assign the final_result column as target feature
y = mainFile['final_result']

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a list of classifiers
classifiers = [
    ('Random Forest', RandomForestClassifier()),
    ('Gradient Boosting', GaussianNB()),
    ('Logistic regression', LogisticRegression(solver='liblinear')),
    ('Support Vector Machine', SVC(kernel='rbf', gamma=0.1, C=1.0)),
    ('K Nearest Neighbour', KNeighborsClassifier())
]

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
# Create a list to store metrics for each classifier
metrics_data = []

for classifier_name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics_data.append([classifier_name, accuracy, precision, recall, f1])

# Convert metrics data to a DataFrame
metrics_df = pd.DataFrame(metrics_data, columns=['Classifier'] + metric_names)
# Create a bar chart for metrics visualization
bar_traces = [
    go.Bar(x=metrics_df['Classifier'], y=metrics_df[metric], name=metric) for metric in metric_names
]
bar_fig = go.Figure(data=bar_traces)


#preview of the dataset

app.layout = html.Div([
    html.H1('Feature Importance Graph'),
    dcc.Graph(figure=fig),
    html.H1('Preview of preprocessed dataset'),
    dash_table.DataTable(data=mainFile.to_dict('records'), page_size=10),
    html.H1('Gender Distribution Pie Chart'),
    dcc.Graph(figure=pie_fig),
    html.H1('Final Result Distribution'),
    dcc.Graph(figure=result_graph),
    html.H1('Comparative studies of Performance metrics'),
    dcc.Graph(figure=bar_fig)

])















