# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('data.csv')
data.shape,

data.head()

data.drop(['Unnamed: 32', 'id'], axis = 1, inplace = True)

data['diagnosis'].value_counts()

x = data.drop('diagnosis', axis = 1)
y = data['diagnosis']
x

def remove_outliers(x):
  for col in x.columns:
    z_score = (x[col]-x[col].mean())/x[col].std()
    if z_score.any()>3 and z_score.any()<-3:
      x.drop(col, axis = 0, inplace = True)
    else:
      x=x
    return x

remove_outliers(x)

def normalize(x):
  scaler  = MinMaxScaler()
  dict = {}
  keys_0=[]
  keys_1=[]
  for col_name in x.columns:
    if data[col_name].describe()['min']>=0 and data[col_name].describe()['max']<=1:
      dict[col_name] = 1
    else:
      dict[col_name] = 0
  for key, val in dict.items():
    if val == 0:
      keys_0.append(key)
      scaled = scaler.fit_transform(x[keys_0])
      df1 = pd.DataFrame(scaled, columns = keys_0)
    else:
      keys_1.append(key)
    df2 = pd.DataFrame(x[keys_1], columns = keys_1)
    df = pd.concat([df1, df2], axis = 1,)
  return df

encoder = LabelEncoder()
y = encoder.fit_transform(y)
x_scaled = normalize(x)


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42)

model = KNeighborsClassifier(n_neighbors = 11)
model.fit(x_train, y_train)
model.predict(x_test)
model.score(x_test, y_test)

model = {
    'KNN':{
        'model': KNeighborsClassifier(),
        'params': {'n_neighbors':[i for i in range(50)]
    }
}
}

model_cvs = GridSearchCV(model['KNN']['model'], model['KNN']['params'])
model_cvs.fit(x_scaled, y)
model_cvs.best_params_

cvs = cross_val_score(KNeighborsClassifier(11), x_scaled, y)

cvs.mean()

neighbors=[]
cv_scores=[]
for k in range(1,52,2):
  neighbors.append(k)
  knn = KNeighborsClassifier(n_neighbors = k)
  scores = cross_val_score(knn, x_scaled,y ,cv = 10, scoring = 'accuracy')
  cv_scores.append(scores.mean())

cv_scores

results = model_cvs.cv_results_
df = pd.DataFrame(results)

df[['param_n_neighbors','mean_test_score']]

mst = df[df.mean_test_score == df.mean_test_score.max()]
optimal_k = int(mst['param_n_neighbors'])

plt.plot(df['param_n_neighbors'], df['mean_test_score'])
plt.xlabel('K Neighbors')
plt.ylabel('Accuracy')
plt.show()

print(classification_report(y_test, model_cvs.predict(x_test)))

