#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import data
pd.set_option("display.max_Columns", None)  #To display all columns 
df = pd.read_csv("./Data/dataset.txt",sep=",")

#Understanding the data
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.dtypes)
print(df.nunique())
print(df["Store ID"].unique())
print(df.groupby("Store ID").ID.nunique())

#Data cleaning
print(df.isna().sum())
df.dropna(inplace=True)
print(df.head())

df["Store ID"]=df["Store ID"].astype("str")

#Data visualization
#Comparation graphics
#Grafico que compara la cantidad de unidades vendidas por cada tienda (las 10 tiendas con mas ventas)
store = df.groupby("Store ID")["Units Sold"].sum()
store_data=store.sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
plt.bar(store_data.index, store_data, width=0.50)
plt.title("Units sold per store")
plt.show()

#Grafico que compara la cantidad de ventas por cada tienda (las 10 tiendas con mas ventas)

count_sales =df.groupby("Store ID")["ID"].count()
count_sales_data=count_sales.sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
plt.plot(count_sales_data.index, count_sales_data, color="red", marker="o")
plt.show()

#Grafico circular con cantidad de ordeenes por cada tiennda 10
sales_store_data = df.groupby("Store ID").count()["ID"].sort_values(ascending=False).head(10)
print(sales_store_data)
plt.figure(figsize=(12,6))
plt.title("Orders per store")
plt.pie(sales_store_data, labels=sales_store_data.index, autopct='%1.1f%%', startangle=90)
plt.show()

#Promedio de ventas por cada tienda
mean_sales = df.groupby("Store ID")["Units Sold"].mean()
mean_sales_data = mean_sales.sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
plt.title("Mean sales per store")
plt.plot(mean_sales_data.index, mean_sales_data, color="green", marker="o")
plt.show()

#Suma de ventas netas
net_sales = df
net_sales["Net Sales"] = net_sales["Units Sold"] * net_sales["Total Price"]
net_sales = net_sales.groupby("Store ID")["Net Sales"].sum()
net_sales_data=net_sales.sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
plt.title("Net sales per store")
plt.bar(net_sales_data.index, net_sales_data, width=0.50)
plt.show()

#Charts with segmented data
#Scatter plot
df['Total Price Grouped'] = (df['Total Price'] // 50) * 50

plt.figure(figsize=(12,6))
plt.scatter(df["Total Price Grouped"], y=df["Units Sold"], alpha=0.5)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df["Total Price"], bins=10, edgecolor='k', alpha=0.6)
plt.show()