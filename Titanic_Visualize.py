import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

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
x = train_d[train_d['Sex'] == 'male']['Survived'].value_counts().tolist()
men_survived = train_d[train_d['Sex']=='male' & train_d['Survived'] == true]
men_not_survived = train_d.shape[0] - men_survived
plt.bar(["men_survived", "men_not_survived"], [men_survived, men_not_survived])
plt.show()
