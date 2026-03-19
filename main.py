import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("data.csv")
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna()
df = df[[
    'price',
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'floors',
    'waterfront',
    'view',
    'condition',
    'sqft_above',
    'sqft_basement',
    'yr_built',
    'yr_renovated',
    'city'
]]
df['house_age'] = 2024 - df['yr_built']
df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
top_cities = df['city'].value_counts().nlargest(10).index
df['city'] = df['city'].apply(lambda x: x if x in top_cities else 'other')
df = pd.get_dummies(df, columns=['city'], drop_first=True)
df = df[df['price'] < df['price'].quantile(0.99)]
df = df.reset_index(drop=True)
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(
    n_estimators=600,
    max_depth=30,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
sample = X_test.iloc[[0]]
prediction = model.predict(sample)
city = "Unknown"
for col in sample.columns:
    if col.startswith("city_") and sample.iloc[0][col] == 1:
        city = col.replace("city_", "")
        break
print("\n HOUSE PRICE PREDICTION REPORT")
print("=" * 50)
print(f"{'City':<15}: {city}")
important_features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'house_age']
for col in important_features:
    if col in sample.columns:
        print(f"{col:<15}: {sample.iloc[0][col]}")
print("\n Estimated Price:")
print(f"{'Price':<15}: ${prediction[0]:,.2f}")
print("=" * 50)