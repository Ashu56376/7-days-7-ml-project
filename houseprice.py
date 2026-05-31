import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Dataset load karo
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

# Train/Test split
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model train karo
model = LinearRegression()
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
print("R² Score:", round(r2_score(y_test, y_pred), 2))
print("RMSE:", round(mean_squared_error(y_test, y_pred)**0.5, 2))
print("Model trained successfully!")
# Apna ghar predict karo!
new_house = pd.DataFrame([{
    'MedInc': 5.0,       # Income (lakhs mein)
    'HouseAge': 20,      # Ghar kitne saal purana
    'AveRooms': 6,       # Total rooms
    'AveBedrms': 2,      # Bedrooms
    'Population': 1500,  # Area ki population
    'AveOccup': 3,       # Logon ki average
    'Latitude': 34.0,
    'Longitude': -118.0
}])

price = model.predict(new_house)
print(f"\nTumhare ghar ki predicted price: ${price[0]*100000:,.0f}")
# Graph banao
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', linewidth=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('result.png')
plt.show()
print("Graph save ho gaya - result.png!")