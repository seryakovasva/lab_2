import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D


train = pd.read_csv('./train11.csv')
#x_test = pd.read_csv('./test.csv')
x_test = train.sample(frac = 0.2)
x_train = pd.concat([train, x_test, x_test]).drop_duplicates(keep=False)
#y_test1 = pd.read_csv('./gender_submission.csv')
#y_test1 = y_test1.drop(['PassengerId'], axis=1)

#x_train = x_train[1:445].reset_index(drop=True)
#x_test = x_train[445:890].reset_index(drop=True)
#x_train, x_test, y_train, y_test = train_test_split(train, train, test_size=0.20, random_state=42)

#зависимая переменная
y_train = x_train['Survived']

#убираем ненужные значения
x_train = x_train.drop(['Name','Ticket','Cabin','PassengerId', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Survived'], axis=1)
x_test = x_test.drop(['Name','Ticket','Cabin','PassengerId', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Survived'], axis=1)
#дополняем значение возраста
x_train['Age'] = x_train.Age.fillna(x_train['Age'].median())
x_train['Age'] = x_train['Age'].astype(int)
#x_train['Fare'] = x_train['Fare'].astype(int)
x_train['Sex'] = x_train['Sex'].map({"male": 0, "female": 1})
x_test['Age'] = x_test.Age.fillna(x_test['Age'].median())
x_test['Age'] = x_test['Age'].astype(int)
#x_test['Fare'] = x_test.Fare.fillna(x_test['Fare'].median())
#x_test['Fare'] = x_test['Fare'].astype(int)
x_test['Sex'] = x_test['Sex'].map({"male": 0, "female": 1})

#
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(svm.SVC(random_state = 30), param_grid, cv=5)
#Обучение
grid_search.fit(x_train, y_train)
grid_search.score(x_train, y_train)
#Распознавание
predicted = grid_search.best_estimator_.predict(x_test)
#результат выживания
y_test = pd.Series(predicted)
print("Наилучшие значения параметров: {}".format(grid_search.best_params_))
print("Наилучшее значение кросс-валидац. правильности:{:.2f}".format(grid_search.best_score_))
print("Правильность на тестовом наборе: {:.2f}".format(grid_search.score(x_test, y_test)))

x_test['Survived'] = pd.Series(predicted, index=x_test.index)

aliveMen = x_test.copy()
aliveWomen = x_test.copy()
deadMen = x_test.copy()
deadWomen = x_test.copy()
aliveMen = aliveMen[(aliveMen['Sex'] == 0) & (aliveMen['Survived'] == 1)]
aliveWomen = aliveWomen[(aliveWomen['Sex'] == 1) & (aliveWomen['Survived'] == 1)]
deadMen = deadMen[(deadMen['Sex'] == 0) & (deadMen['Survived'] == 0)]
deadWomen = deadWomen[(deadWomen['Sex'] == 1) & (deadWomen['Survived'] == 0)]
print('----------')
print(aliveMen)
print('----------')
print(aliveWomen)
print('----------')
print(deadMen)
print('----------')
print(deadWomen)

#группировка по возрасту
x_test = x_test.sort_values(by = ['Age'])
r = [-1, 16, 35, 45, 65, 120]
g = [0, 1, 2, 3, 4]
x_test['Age'] = pd.cut(x_test['Age'], bins=r, labels=g)

#x_test.pivot_table('Age', ['Survived', 'Sex'], 'Pclass', 'count').plot(kind='bar', stacked=True)
x_test.pivot_table('Pclass', ['Sex', 'Age'], 'Survived', 'count').plot(kind='bar', stacked=True)
plt.ylabel('count survived')
plt.xlabel('Sex, Age')
plt.show()

