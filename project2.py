import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Check if necessary packages are installed
required_packages = ['numpy', 'pandas', 'sklearn', 'matplotlib']
for package in required_packages:
    if package not in sys.modules:
        print(f"Error: {package} is not installed. Please install it using 'pip install {package}'")
        sys.exit(1)

# Sample data: Area (in square feet) and corresponding Flat Prices (in thousands)
data = {
    'Area': [500, 700, 1000, 1200, 1500, 1800, 2000],
    'Price': [50, 70, 100, 120, 150, 180, 200]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Prepare the data for the model
X = df[['Area']]  # Independent variable (Area)
y = df['Price']    # Dependent variable (Price)

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict flat prices for new areas
new_areas = np.array([600, 800, 1600]).reshape(-1, 1)
predicted_prices = model.predict(new_areas)

# Output the predictions
for area, price in zip(new_areas.flatten(), predicted_prices):
    print(f"Predicted price for {area} sq ft: {price:.2f}k")

# Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(new_areas, predicted_prices, color='green', marker='x', label='Predictions')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (k)')
plt.title('Flat Price Prediction Based on Area')
plt.legend()
plt.show()
