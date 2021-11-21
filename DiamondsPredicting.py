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
test_size = 25
random_state = 202
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)

# Тренировка
regr = RandomForestRegressor(n_estimators = 10, max_depth = 10, random_state = random_state)
regr.fit(x_train, y_train.values.ravel())

# Прогнозрование
predictions = regr.predict(x_test)

result               = x_test
result['price']      = y_test
result['prediction'] = predictions.tolist()
result['deviation']  = 100 * (result['price'] - result['prediction']) / result['price']
print(result.to_string())
print('deviation min,max: ', '{:.2f}'.format(min(abs(result['deviation']))), '{:.2f}'.format(max(abs(result['deviation']))))

# Определение оси X
x_axis = x_test.carat

# Построение графика
plt.scatter(x_axis, y_test,      c = 'b', alpha = 0.5, marker = '.', label = 'Real')
plt.scatter(x_axis, predictions, c = 'r', alpha = 0.5, marker = '.', label = 'Predicted')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.grid(color = '#d3d3d3', linestyle = 'solid')
plt.legend(loc = 'lower right')
plt.show()
