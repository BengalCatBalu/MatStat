import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Probit

data = pd.read_excel('data.xlsx')
# делим данные на обучающие и тестовые
train_data = data.iloc[:961]
test_data = data.iloc[961:]
# Создаем dummy переменные
data['age_22'] = np.where(data['age'] < 22 * 12, 1, 0)
data['age_30'] = np.where(data['age'] > 30 * 12, 1, 0)
data['rules_5'] = np.where(data['rules'] < 5, 1, 0)
data['rules_10'] = np.where(data['rules'] > 10, 1, 0)

# делим данные на обучающие и тестовые
train_data = data.iloc[:961]
test_data = data.iloc[961:]

# Выбираем столбцы, которые хотим использовать в качестве регрессоров
X = train_data[['age_22', 'age_30', 'rules_5', 'rules_10', 'black', 'married', 'drugs']]
y = train_data['recid']
X_test = test_data[['age_22', 'age_30', 'rules_5', 'rules_10', 'black', 'married', 'drugs']]



# Создаем линейную регрессию
#lin_reg = LinearRegression()
#linear_result = lin_reg.fit(X, y)

# Вывод оцененных коэффициентов регрессии
# print("Оцененные коэффициенты для линейной регрессии".upper())
# coefficients = lin_reg.coef_
# for i, coef in enumerate(coefficients):
    # print(f'| {X.columns[i]} | {coef} |')

# Предсказывание данных
# y_pred = lin_reg.predict(X_test)
# print("Предсказанные данные".upper())
# for x in y_pred:
    # print(x)




# cоздаем логистическую регрессию
#log_reg = LogisticRegression()
#log_reg.fit(X, y)
# Предсказание для тестовых данных
#y_pred = log_reg.predict(X_test)
#print("Оцененные коэффициенты для логит. регрессии".upper())
#coeff = log_reg.coef_
#for i in range(len(coeff[0])):
    #print(f'{coeff[0][i]}')
print("Предсказанные данные".upper())
#for x in y_pred:
    #print(x)

# Создайте модель
logit_model=sm.Logit(y,X)

# Обучите модель
result=logit_model.fit()
cov_matrix = result.cov_params()
print("Ковариационная матрица".upper())
print(cov_matrix)
#Выведите результаты

print(result.summary())
for x in result.predict(X_test):
    print(x)

# Создайте экземпляр модели
probit_model = Probit(y, X)

# Обучите модель
result = probit_model.fit()
cov_matrix = result.cov_params()
print("Ковариационная матрица".upper())
print(cov_matrix)
print(result.summary())

for x in result.predict(X_test):
    print(x)
