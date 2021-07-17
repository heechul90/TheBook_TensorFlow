### 필요 라이브러리 호출
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


### 데이터 준비
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv('./chapter03/data/iris.data', names=names)
dataset


### 훈련셋, 테스트셋으로 분리
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X_train = s.fit(X_train).transform(X_train)
X_test = s.fit(X_test).transform(X_test)



### 모델 생성 및 훈련
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)


### 모델 정확도
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print('정확도 : {}'.format(accuracy_score(y_test, y_pred)))


### 최적의 K 찾기
k = 10
acc_array = np.zeros(k)
for k in np.arange(1, k+1, 1):
    classfier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred = classfier.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k-1] = acc

max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print('정확도', max_acc, ' 으로 최적의 K는', k+1, ' 입니다.')



