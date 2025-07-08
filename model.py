import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load the data
data = pd.read_csv("data/train.csv")

# 2. Select features and target
features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF",
            "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = data[features]
y = data["SalePrice"]

# 3. Split into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 4. Train Decision Tree model
dt_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
dt_model.fit(train_X, train_y)
dt_preds = dt_model.predict(val_X)
dt_mae = mean_absolute_error(val_y, dt_preds)

# 5. Train Random Forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_preds = rf_model.predict(val_X)
rf_mae = mean_absolute_error(val_y, rf_preds)

print("📊 Decision Tree MAE:", dt_mae)
print("🌲 Random Forest MAE:", rf_mae)

# 6. Fit final model on all data
final_model = RandomForestRegressor(random_state=1)
final_model.fit(X, y)

# 7. Load test data and predict
test_data = pd.read_csv("data/test.csv")
test_X = test_data[features]
test_preds = final_model.predict(test_X)

# 8. Save predictions
output = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': test_preds
})
output.to_csv('submission.csv', index=False)
print("✅ Submission file 'submission.csv' created.")
