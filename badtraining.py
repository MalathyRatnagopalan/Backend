import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# CSV-Datei laden
file_path = 'cleaned_rent_data.csv'  # Pfad zur bereinigten CSV-Datei
data = pd.read_csv(file_path)

# Vorverarbeitung der Daten
# One-Hot-Encoding für kategoriale Daten
X = pd.get_dummies(data.drop(columns=['Price']))
y = data['Price']

# Aufteilen der Daten in Trainings-, Validierungs- und Testsets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Modell 1: Lineare Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_val_pred = lin_reg.predict(X_val)
lin_reg_test_pred = lin_reg.predict(X_test)

# Modell 2: Decision Tree Regressor mit angepassten Hyperparametern
tree_reg = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
tree_reg.fit(X_train, y_train)
tree_reg_val_pred = tree_reg.predict(X_val)
tree_reg_test_pred = tree_reg.predict(X_test)

# Leistungsmetriken für das Validierungsset
lin_reg_val_mae = mean_absolute_error(y_val, lin_reg_val_pred)
lin_reg_val_mse = mean_squared_error(y_val, lin_reg_val_pred)
lin_reg_val_rmse = mean_squared_error(y_val, lin_reg_val_pred, squared=False)

tree_reg_val_mae = mean_absolute_error(y_val, tree_reg_val_pred)
tree_reg_val_mse = mean_squared_error(y_val, tree_reg_val_pred)
tree_reg_val_rmse = mean_squared_error(y_val, tree_reg_val_pred, squared=False)

# Leistungsmetriken für das Testset
lin_reg_test_mae = mean_absolute_error(y_test, lin_reg_test_pred)
lin_reg_test_mse = mean_squared_error(y_test, lin_reg_test_pred)
lin_reg_test_rmse = mean_squared_error(y_test, lin_reg_test_pred, squared=False)

tree_reg_test_mae = mean_absolute_error(y_test, tree_reg_test_pred)
tree_reg_test_mse = mean_squared_error(y_test, tree_reg_test_pred)
tree_reg_test_rmse = mean_squared_error(y_test, tree_reg_test_pred, squared=False)

# Ausgabe der Ergebnisse
print("Validierungsergebnisse Lineare Regression:")
print("Validierungs-MAE:", lin_reg_val_mae)
print("Validierungs-MSE:", lin_reg_val_mse)
print("Validierungs-RMSE:", lin_reg_val_rmse)

print("\nValidierungsergebnisse Decision Tree Regressor:")
print("Validierungs-MAE:", tree_reg_val_mae)
print("Validierungs-MSE:", tree_reg_val_mse)
print("Validierungs-RMSE:", tree_reg_val_rmse)

print("\nTestergebnisse Lineare Regression:")
print("Test-MAE:", lin_reg_test_mae)
print("Test-MSE:", lin_reg_test_mse)
print("Test-RMSE:", lin_reg_test_rmse)

print("\nTestergebnisse Decision Tree Regressor:")
print("Test-MAE:", tree_reg_test_mae)
print("Test-MSE:", tree_reg_test_mse)
print("Test-RMSE:", tree_reg_test_rmse)