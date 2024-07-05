import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

sales = pd.read_csv('home_data.csv')
np.random.seed(416)

sales = sales.sample(frac=0.01, random_state=0) 

selected_inputs = [
    'bedrooms', 
    'bathrooms',
    'sqft_living', 
    'sqft_lot', 
    'floors', 
    'waterfront', 
    'view', 
    'condition', 
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built', 
    'yr_renovated'
]

all_features = []
for feature_name in selected_inputs:
    squared_feature_name = feature_name + '_square'
    sqrt_feature_name = feature_name + '_sqrt'
    
    sales[squared_feature_name] = sales[feature_name] ** 2
    sales[sqrt_feature_name] = sales[feature_name] ** (1/2)

    all_features.extend([feature_name, squared_feature_name, sqrt_feature_name])

price = sales['price']
sales = sales[all_features]

sales_train_and_validation, sales_test, price_train_and_validation, price_test = \
    train_test_split(sales, price, test_size=0.2, random_state=6)
sales_train, sales_validation, price_train, price_validation = \
    train_test_split(sales_train_and_validation, price_train_and_validation, test_size=.125, random_state=6)

scaler = StandardScaler()
scaler.fit(sales_train)

sales_train_standardized = scaler.transform(sales_train)
sales_validation_standardized = scaler.transform(sales_validation)
sales_test_standardized = scaler.transform(sales_test)

linear = LinearRegression()
linear.fit(sales_train_standardized, price_train)
rmse_test_unregularized = mean_squared_error(linear.predict(sales_test_standardized), price_test, squared=False)

l2_lambdas = np.logspace(-5, 5, 11, base=10)
ridge_data = []
for i in range(len(l2_lambdas)):
    ridge_model = Ridge(alpha=l2_lambdas[i])
    ridge_model.fit(sales_train_standardized, price_train)

    ridge_data.append({
        'l2_penalty': l2_lambdas[i],
        'model': ridge_model,
        'rmse_train': mean_squared_error(ridge_model.predict(sales_train_standardized), price_train, squared=False),
        'rmse_validation': mean_squared_error(ridge_model.predict(sales_validation_standardized), price_validation, squared=False)
    })
ridge_data = pd.DataFrame(ridge_data)

bestseries = ridge_data.loc[ridge_data["rmse_validation"].idxmin()]
best_l2 = bestseries["l2_penalty"]
rmse_test_ridge = mean_squared_error(bestseries["model"].predict(sales_test_standardized), price_test, squared=False)
num_zero_coeffs_ridge = sum(coef == 0 for coef in bestseries["model"].coef_)

l1_lambdas = np.logspace(1, 7, 7, base=10)
lasso_data = []
for i in range(len(l1_lambdas)):
    lasso_model = Lasso(alpha=l1_lambdas[i])
    lasso_model.fit(sales_train_standardized, price_train)

    lasso_data.append({
        'l1_penalty': l1_lambdas[i],
        'model': lasso_model,
        'rmse_train': mean_squared_error(lasso_model.predict(sales_train_standardized), price_train, squared=False),
        'rmse_validation': mean_squared_error(lasso_model.predict(sales_validation_standardized), price_validation, squared=False)
    })
lasso_data = pd.DataFrame(lasso_data)

bestseriesl1 = lasso_data.loc[lasso_data["rmse_validation"].idxmin()]
best_l1 = bestseriesl1["l1_penalty"]
rmse_test_lasso = mean_squared_error(bestseriesl1["model"].predict(sales_test_standardized), price_test, squared=False)
num_zero_coeffs_lasso = sum(coef == 0 for coef in bestseriesl1["model"].coef_)

best_model_lasso = bestseriesl1['model']

zero_coef_features = []
nonzero_coef_features = []
for feature, coef in zip(all_features, best_model_lasso.coef_):
    if abs(coef) <= 10 ** -17:
        zero_coef_features.append(feature)
    else:
        nonzero_coef_features.append(feature)

print('L2 Penalty',  best_l2)
print('Test RMSE (Ridge)', rmse_test_ridge)
print('Num Zero Coeffs (Ridge)', num_zero_coeffs_ridge)
print('Best L1 Penalty', best_l1)
print('Test RMSE (Lasso)', rmse_test_lasso)
print('Num Zero Coeffs (Lasso)', num_zero_coeffs_lasso)
print("Features with coefficient == 0:", zero_coef_features)
print("Features with coefficient != 0:", nonzero_coef_features)
