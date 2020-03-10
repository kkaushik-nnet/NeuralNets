import pandas as pd
col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima = pd.read_csv("C:/Users/Kaushik/Documents/Kaggle Datasets/Classification Datasets/Diabetes/diabetes.csv", names = col_names)

pima.head()

# Define X and y
feature_cols = ['pregnant','insulin','bmi','age']
X = pima[feature_cols]
y = pima.label

# Split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

# Train a Logistic Regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train,y_train)

# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)

                            # Classification Accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred_class))

               # Null Accuracy : accuracy that could be achieved by always predicting the most frequent class
# Examine the class distribution of the testing set (using Panda series method)
y_test.value_counts()

# Calculate percentage of ones
y_test.mean()

# Calculate percentage of zeros
1 - y_test.mean()

# Calculate Null Accuracy
y_test.value_counts().head(1)/len(y_test)

# Comparing the true and predicted responses
print('True',y_test.values[0:25])
print('Pred',y_pred_class[0:25])

#Confusion matrix: First arg is True values , Second is Predicted values
metrics.confusion_matrix(y_test,y_pred_class)

# Save confusion matrix and slice it into four pieces
confusion = metrics.confusion_matrix(y_test,y_pred_class)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]

Classificaton_Accuracy = (TP+TN)/(TP+TN+FP+FN) # metrics.accuracy_score()
Classification_Error = 1-Classificaton_Accuracy
Sensitivity = TP/(TP+FN) # metrics.recall_score()
Specificity = TN/(TN+FP)
Precision = TP/(TP+FP) # metrics.precision_score()