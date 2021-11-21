import numpy as np
import pandas as pd
import seaborn as anb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.ensemble        import RandomForestRegressor

df = pd.read_csv('P1_diamonds.csv')
# print(df.head(10).to_string())

# Удаление Unnamed столбца
df = df.drop(['Unnamed: 0'], axis = 1)
# print(df.head(10).to_string())

# Создание переменных для категорий
categorical_features = ['cut','color','clarity']
le = LabelEncoder()

# Замена категорий на численные значения
for i in range(3):
  new = le.fit_transform(df[categorical_features[i]])
  df[categorical_features[i]] = new
# print(df.head(10).to_string())

x = df[['carat','cut','color','clarity','depth','table','x','y','z']]
y = df[['price']]

# Разделение данных на тренировочный и тестовый наборы
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 15, random_state = 101)

# Тренировка
regr = RandomForestRegressor(n_estimators = 10, max_depth = 10, random_state = 101)
regr.fit(x_train, y_train.values.ravel())

# Прогнозрование
predictions = regr.predict(x_test)

result = x_test
result['price'] = y_test
result['prediction'] = predictions.tolist()

print(result.to_string())
