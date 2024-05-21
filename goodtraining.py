import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

# CSV-Datei laden
file_path = 'cleaned_rent_data.csv'
data = pd.read_csv(file_path)

# Vorverarbeitung der Daten
X = pd.get_dummies(data.drop(columns=['Price']))
y = data['Price']

# Aufteilen der Daten in Trainings-, Validierungs- und Testsets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Pipeline für Feature Scaling und Modelltraining
pipeline_lin_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

pipeline_tree_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Modelltraining
pipeline_lin_reg.fit(X_train, y_train)
pipeline_tree_reg.fit(X_train, y_train)

# Vorhersagen und Metriken für das Validierungsset
lin_reg_val_pred = pipeline_lin_reg.predict(X_val)
tree_reg_val_pred = pipeline_tree_reg.predict(X_val)

lin_reg_val_mae = mean_absolute_error(y_val, lin_reg_val_pred)
lin_reg_val_mse = mean_squared_error(y_val, lin_reg_val_pred)
lin_reg_val_rmse = mean_squared_error(y_val, lin_reg_val_pred, squared=False)

tree_reg_val_mae = mean_absolute_error(y_val, tree_reg_val_pred)
tree_reg_val_mse = mean_squared_error(y_val, tree_reg_val_pred)
tree_reg_val_rmse = mean_squared_error(y_val, tree_reg_val_pred, squared=False)

# Vorhersagen und Metriken für das Testset
lin_reg_test_pred = pipeline_lin_reg.predict(X_test)
tree_reg_test_pred = pipeline_tree_reg.predict(X_test)

lin_reg_test_mae = mean_absolute_error(y_test, lin_reg_test_pred)
lin_reg_test_mse = mean_squared_error(y_test, lin_reg_test_pred)
lin_reg_test_rmse = mean_squared_error(y_test, lin_reg_test_pred, squared=False)

tree_reg_test_mae = mean_absolute_error(y_test, tree_reg_test_pred)
tree_reg_test_mse = mean_squared_error(y_test, tree_reg_test_pred)
tree_reg_test_rmse = mean_squared_error(y_test, tree_reg_test_pred, squared=False)

# Ausgabe der Ergebnisse für Validierungs- und Testsets
print("Validierungsergebnisse Lineare Regression:")
print("MAE:", lin_reg_val_mae)
print("MSE:", lin_reg_val_mse)
print("RMSE:", lin_reg_val_rmse)

print("\nValidierungsergebnisse Decision Tree Regressor:")
print("MAE:", tree_reg_val_mae)
print("MSE:", tree_reg_val_mse)
print("RMSE:", tree_reg_val_rmse)

print("\nTestergebnisse Lineare Regression:")
print("MAE:", lin_reg_test_mae)
print("MSE:", lin_reg_test_mse)
print("RMSE:", lin_reg_test_rmse)

print("\nTestergebnisse Decision Tree Regressor:")
print("MAE:", tree_reg_test_mae)
print("MSE:", tree_reg_test_mse)
print("RMSE:", tree_reg_test_rmse)