### 필요한 라이브러리 호출
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='darkgrind')


### 데이터 호출
cols = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety','output']
cars = pd.read_csv('../chap2/data/car_evaluation.csv', names=cols, header=None)