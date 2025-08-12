import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

sales_df=pd.read_csv("D:\DBMS PROJECT\Sales and Inventory_sales.csv")
products_df=pd.read_csv("D:\DBMS PROJECT\Sales and Inventory_products.csv")
customers_df=pd.read_csv("D:\DBMS PROJECT\Sales and Inventory_customers.csv")
merged_df = pd.merge(sales_df, products_df, on="product_id")
merged_df = pd.merge(merged_df, customers_df, on="customer_id")
merged_df.head()

merged_df=merged_df.dropna()

fig=px.scatter(merged_df,x="Quantity",y="TotalAmount",size="Quantity")
fig.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(sales_df['Quantity'], bins=20, kde=True)
plt.title('Histogram of Quantity Purchased')

plt.subplot(2, 2, 2)
sns.histplot(sales_df['TotalAmount'], bins=20, kde=True)
plt.title('Histogram of Total Amount')

print("Products DataFrame Columns:", products_df.columns)
print("Customers DataFrame Columns:", customers_df.columns)

# Plot bar plot for customer countries
plt.subplot(1, 2, 2)
sns.countplot(data=customers_df, x='State')
plt.title('Number of Customers by State')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(sales_df['Quantity'], bins=20, kde=True)
plt.title('Histogram of Quantity Purchased')

plt.subplot(2, 2, 2)
sns.histplot(sales_df['TotalAmount'], bins=20, kde=True)
plt.title('Histogram of Total Amount')

print("Products DataFrame Columns:", products_df.columns)
print("Customers DataFrame Columns:", customers_df.columns)

# Plot bar plots for product categories
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=products_df, x='Product Name')
plt.title('Number of Products by Category')
plt.xticks(rotation=45)

# Plot bar plot for customer countries
plt.subplot(1, 2, 2)
sns.countplot(data=customers_df, x='State')
plt.title('Number of Customers by State')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

label_encoder = LabelEncoder()
merged_df['state_encoded'] = label_encoder.fit_transform(merged_df['State'])
merged_df['city_encoded'] = label_encoder.fit_transform(merged_df['City'])
X = merged_df[['product_id', 'customer_id', 'ProductUnitPrice', 'Discount', 'StockQuantity', 'state_encoded', 'city_encoded']]
y = merged_df['Quantity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)

joblib.dump(model, 'trained_model.pkl')
loaded_model = joblib.load('trained_model.pkl')

plt.figure(figsize=(10, 6))
plt.scatter(X_test.index, y_test, color='blue', label='Actual')  # Use index as x-values
plt.scatter(X_test.index, y_pred, color='red', label='Predicted')  # Use index as x-values
plt.title('Actual vs. Predicted Data')
plt.xlabel('Index')
plt.ylabel('Quantity')
plt.legend()
plt.show()




