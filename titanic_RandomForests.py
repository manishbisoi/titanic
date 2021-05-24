import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


sns.color_palette("mako", as_cmap=True)
"""
List of Columns along with data types
PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
"""

train_d = pd.read_csv('./train.csv', sep=',')
test_d = pd.read_csv('./test.csv', sep=',')

train_d = train_d.dropna(how='any', axis = 0)
features = ['Age', 'SibSp', 'Parch', 'Fare']
X = train_d[features]
y = train_d['Survived']

"""
irrelevant_features = ['Name', 'Ticket', 'Cabin','PassengerId','Sex','Embarked']
train_d = train_d.drop(irrelevant_features, axis = 1)
test_d = test_d.drop(irrelevant_features, axis = 1)
train_d = train_d.drop(['Survived'], axis=1)

train_d = train_d.dropna(axis = 0, how='any')
test_d = test_d.dropna(axis=0, how='any')
#train_d.to_csv('TEMP__TrainingData.csv')
#test_d.to_csv('TEMP__TestingData.csv')
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
reg = RandomForestRegressor(n_estimators = 40, random_state=42)
reg.fit(X_train, y_train)
predictions = []
for each in reg.predict(X_test):
    if each > 0.5 : predictions.append(1)
    else : predictions.append(0)

#print(reg.score(X_train, y_train))
#print(mean_absolute_error( reg.predict(X_test), y_test ))
#print(predictions, y_test)
#print(accuracy_score(reg.predict(X_test), y_test))
cf_matrix = confusion_matrix(y_test, predictions)

#Seaborn heatmap plot
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='')
#plt.show()
plt.title("Random Forest Predictions")
plt.savefig("RandomForest_Pred_HeatMap.jpg")

#print(type(predictions), type(y_test.tolist()))

#df = pd.DataFrame({'Predictions':predictions, 'Actuals':y_test})
#df.to_csv("Predictions and Actuals.csv")
