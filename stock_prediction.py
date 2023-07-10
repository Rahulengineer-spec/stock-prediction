import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('stock_data.csv')

# Convert the 'Date' column to a pandas datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Sort the data by date in ascending order
df.sort_values(by='Date', inplace=True)

# Reset the index
df.reset_index(inplace=True, drop=True)
      
# Determine the index to split the data
split_index = int(0.8 * len(df))

# Split the data into training and testing sets
train_data = df[:split_index]
test_data = df[split_index:]

# Extract the 'Close' prices as the labels
train_labels = train_data['Close'].values
test_labels = test_data['Close'].values

# Extract the numerical date representation as the features
train_features = pd.to_numeric(train_data['Date']).values.reshape(-1, 1)
test_features = pd.to_numeric(test_data['Date']).values.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(train_features, train_labels)

# Make predictions on the test data
predictions = model.predict(test_features)

# Plot the actual prices and predicted prices
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'], test_labels, label='Actual')
plt.plot(test_data['Date'], predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.show()
