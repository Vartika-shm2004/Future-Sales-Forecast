# Line chart showing real sales vs model predictions
plt.figure(figsize=(12, 5))
plt.plot(data.index, data['sales'], label='Actual', marker='o')
plt.plot(X_test.index, y_pred, label='Predicted', marker='x')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()

# Check prediction errors distribution
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel('Prediction Error ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.show()


# Average sales by month (seasonality)
monthly_avg = data.groupby('month')['sales'].mean()
plt.figure(figsize=(10, 5))
plt.bar(monthly_avg.index, monthly_avg.values, color='steelblue')
plt.xticks(range(1, 13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
plt.xlabel('Month')
plt.ylabel('Average Sales ($)')
plt.title('Seasonal Sales Pattern')
plt.show()
