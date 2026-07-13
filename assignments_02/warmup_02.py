import numpy as np
from sklearn.linear_model import LinearRegression

#scikit learn Q1
years = np.array([1,2,3,5,7,10]).reshape(-1,1)
salary = np.array([45000,50000,60000,75000,90000,120000])

model = LinearRegression()
model.fit(years,salary) #relationship b/n years & salary
predict1 = model.predict([[4]])
predict2 = model.predict([[8]])

print(f"Slope:\n {model.coef_[0]}")
print(f"Intercept:\n {model.intercept_}")
print(f"Prediction 1:\n {predict1}")
print(f"Prediction 2:\n {predict2}")

#scikit learn Q2
x = np.array([10,20,30,40,50])
new_x = np.array([10,20,30,40,50]).reshape(-1,1)

print(f"1D shape:\n {x.shape}")
print(f"2D shape:\n {new_x.shape}")

#Leaving x as an 1D array will cause errors when calling .fit() and .predict()
#Making x as 2D allows for complex computation later on.