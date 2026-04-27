# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read dataset, clean column names, and separate input features (Size, Bedrooms) and target values (Price, Occupants).
2. Normalize the input features using standard scaling.
3. Train two SGD regression models using the scaled data to predict Price and Occupants.
4. Take user input, scale it, predict outputs using both models, and display the results. 

## Program:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

df=pd.read_csv("house.csv")
df.columns=df.columns.str.strip()
x=df[["Size","Bedrooms"]]
y_price=df["Price"]
y_occ=df["Occupants"]

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

price_model=SGDRegressor(max_iter=1000,learning_rate='constant',eta0=0.01)
occ_model=SGDRegressor(max_iter=1000,learning_rate='constant',eta0=0.01)
price_model.fit(x_scaled,y_price)
occ_model.fit(x_scaled,y_occ)

size=float(input("Size of the House: "))
bed=int(input("No. of Beds in the House: "))
new_data=scaler.transform([[size,bed]])
price_pred=price_model.predict(new_data)
occ_pred=occ_model.predict(new_data)


print("Predicted Price: ",price_pred)
print("Predicted Occupants: ",round(occ_pred[0]))
```

## Output:

<img width="367" height="92" alt="image" src="https://github.com/user-attachments/assets/93104088-61a1-4ce4-838f-477eb6256d2e" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
