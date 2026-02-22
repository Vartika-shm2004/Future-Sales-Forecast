import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
print("Generating Sample Data....")


np.random.seed(42)
dates=pd.date_range(start='2022-01-01', end='2024-12-01', freq='MS')
n=len(dates)
#DATA GENERATION
data=pd.DataFrame({
    'date': dates,
    'month': dates.month,
    'year':dates.year,
    'quarter': dates.quarter,
    'day_of_year': dates.dayofyear,
    'promotion': np.random.choice([0,1],n,p=[0.6,0.4]),
    'holiday_season': dates.month.isin([11,12]).astype(int),
    'competitor_price': np.random.uniform(50,150,n),
    'advertising_budget': np.random.uniform(1000,10000,n),
    'price': np.random.uniform(20,100,n),
    'store_traffic': np.random.randint(100,1000,n)
})
base_sales=500
data['sales']=(
    base_sales
    +200*data['promotion']
    +150*data['holiday_season']
    -0.5*data['competitor_price']
    +0.1*data['advertising_budget']
    -2*data['price']
    +0.5*data['store_traffic']
    +50*np.sin(data['month']*np.pi/6)
    +np.random.normal(0,50,n)
)
data['sales']= data['sales'].clip(lower=0).round(0).astype(int)
print(f" Generated {len(data)} records from {dates[0].date()} to {dates[-1].date()}")
print(f"Sample data:\n {data.head()}")
print(data)

#FEATURE PROCESSING AND TARGET VARIABLES
print("Feature processing and target")
feature_column=['month','year','quarter','day_of_year','promotion','advertising_budget','holiday_season','competitor_price','price','store_traffic']
X=data[feature_column]
Y=data['sales']
print(f"features: {feature_column}")
print(f" target: sales")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

#SPLITING THE DATA
print("Split data into Training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f" Training set: {len(X_train)} samples")
print(f" Testing set: {len(X_test)} samples")

#TRAINING AND TESTING THE MODEL
print("\n Training the model...")
model= GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
print("Model training completed!")

#MODEL EVALUATION 
print("\n Evaluating model performance...")
y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
print("="*50)
print("Model Performance Metrices")
print("="*50)
print(f"Mean Squared Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R2 Score: {r2:,.4f}")

#2025 YEAR SALES FORECASTING
print("\n Forecasting 2025 sales...")
future_dates= pd.date_range(start='2025-01-01', end='2025-12-01', freq='MS')
future=pd.DataFrame({
    'date': future_dates,
    'month': future_dates.month,
    'year': [2025]*12,
    'quarter': future_dates.quarter,
    'day_of_year': future_dates.dayofyear,
    'promotion':[1,0,0,0,0,1,0,0,0,1,1,1],
    'holiday_season':[0,0,0,0,0,0,0,0,0,1,1,1],
    'competitor_price': [80]*12,
    'advertising_budget': [5000]*12,
    'price': [50]* 12,
    'store_traffic': [500]*12})
future_prediction=model.predict(future[feature_column])
future['predicted_sales']=future_prediction.round(0).astype(int)
print("="*50)
print("2025 monthly sales forecast")
print("="*50)
print(future[['date','predicted_sales']].to_string(index=False))
total_2025=future_prediction.sum()
print(f"\n>>>Total 2025 projected sales:${total_2025:,.0f}")

#2026 YEAR SALES FORECASTING
print("\n 2026 Sales Forecasting")
future_dates=pd.date_range(start='2026-01-01', end='2026-12-01', freq='MS')
future=pd.DataFrame({
    'date': future_dates,
    'month': future_dates.month,
    'year': [2026]*12,
    'quarter': future_dates.quarter,
    'day_of_year': future_dates.year,
    'promotion':[1,1,1,1,0,0,0,1,1,0,1,0],
    'holiday_season':[0,0,0,0,0,1,1,1,0,0,0,1],
    'competitor_price':[100]*12,
    'advertising_budget':[8000]*12,
    'price':[80]*12,
    'store_traffic':[500]*12})
future_sales_prediction=model.predict(future[feature_column])
future['predicted_sale']=future_sales_prediction.round(0).astype(int)
print('='*50)
print("2026 monthly sales")
print('='*50)
print(future[['date','predicted_sale']].to_string(index=False))
total_sales=future_sales_prediction.sum()

print(f">>>2026 Total sales: ${total_sales: ,.0f}")
