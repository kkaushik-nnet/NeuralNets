import pandas as pd
import numpy as np
from sklearn import metrics
col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima = pd.read_csv("C:/Users/Kaushik/Documents/Kaggle Datasets/Classification Datasets/Diabetes/diabetes.csv", names = col_names)

pima.head()

# Define X and y
feature_cols = ['pregnant','insulin','bmi','age']
X = pima[feature_cols]
y = pima.label

# Split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, shuffle=True, random_state=0)


# Train a Logistic Regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs').fit(X_train,y_train)

# Decision Tree model
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

#RandomForest
from sklear.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_jobs = 2, random_state=0).fit(train_X, train_y)


# make class predictions for the testing set
y_pred_class_log = logreg.predict(X_test)
y_pred_class_tree = tree_model.predict(X_test)
y_pred_class_rf = rf_model.predict(X_test)
          
#confusion matrix: First arg is True values , Second is Predicted values
metrics.confusion_matrix(y_test,y_pred_class_log)


# Save confusion matrix and slice it into four pieces
confusion_log = metrics.confusion_matrix(y_test,y_pred_class_log)
TP = confusion_log[1,1]
TN = confusion_log[0,0]
FP = confusion_log[0,1]
FN = confusion_log[1,0]

Classificaton_Accuracy = (TP+TN)/(TP+TN+FP+FN) # metrics.accuracy_score()
Classification_Error = 1-Classificaton_Accuracy
Sensitivity = TP/(TP+FN) # metrics.recall_score()
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP) # metrics.precision_score()