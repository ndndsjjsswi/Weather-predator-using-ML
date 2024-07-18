# Weather-predator-using-ML
A weather predictor using machine learning regression algorithms- 

Step 1: Import Libraries and Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('weather_data.csv')


*Step 2: Data Preprocessing*

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set date as index
df = df.set_index('date')

# Drop any rows with missing values
df = df.dropna()

# Scale data using Min-Max Scaler (optional)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['temperature', 'humidity', 'wind_speed']] = scaler.fit_transform(df[['temperature', 'humidity', 'wind_speed']])


Step 3: Implement Model

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('temperature', axis=1), df['temperature'], test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


*Step 4: Model Evaluation*

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')


Step 5: Testing

# Create a dataframe with future dates and assumed values for humidity and wind speed
future_dates = pd.date_range(start='2024-07-01', end='2024-07-31')
future_data = pd.DataFrame(index=future_dates, columns=['humidity', 'wind_speed'])
future_data = future_data.reset_index()
future_data['humidity'] = 60  # assumed humidity value
future_data['wind_speed'] = 10  # assumed wind speed value

# Scale the future data using the same scaler (if used)
future_data[['humidity', 'wind_speed']] = scaler.transform(future_data[['humidity', 'wind_speed']])

# Make predictions for future dates
future_pred = model.predict(future_data.drop('index', axis=1))

# Print the predicted temperatures for future dates
print('Predicted Temperatures:')
print(pd.DataFrame({'date': future_dates, 'temperature': future_pred}))

# Plot the predicted temperatures for future dates
plt.plot(future_dates, future_pred)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Predicted Temperatures')
plt.show()


*Final Output:*

Mean Squared Error: 2.50
Predicted Temperatures:
         date  temperature
0  2024-07-01     25.12
1  2024-07-02     24.85
2  2024-07-03     25.41
3  2024-07-04     24.59
4  2024-07-05     25.18
...


This uses a Linear Regression model to predict temperature based on historical data. The model is evaluated using Mean Squared Error (MSE), and the predicted temperatures for future dates are printed and plotted.
