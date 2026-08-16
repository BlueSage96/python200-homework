import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Scikit Learn Q1
years = np.array([1,2,3,5,7,10]).reshape(-1,1)
salary = np.array([45000,50000,60000,75000,90000,120000])

model = LinearRegression()
model.fit(years,salary) #relationship b/n years & salary
predict1 = model.predict([[4]])
predict2 = model.predict([[8]])

print(f"Scikit Learn Q1:\n")
print(f"Slope:\n {model.coef_[0]}")
print(f"Intercept:\n {model.intercept_}")
print(f"Prediction 1:\n {predict1}")
print(f"Prediction 2:\n {predict2}")

# Scikit Learn Q2
'''
Scikit-learn expects X to be a 2D feature matrix, where each row is an
observation and each column is a feature. The original x array is 1D,
so reshape(-1,1) changes it into a 2D array with one feature column.
'''

x = np.array([10,20,30,40,50])
new_x = np.array([10,20,30,40,50]).reshape(-1,1)

print(f"\nScikit Learn Q2:\n")
print(f"1D shape:\n {x.shape}")
print(f"2D shape:\n {new_x.shape}")

# Scikit Learn Q3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print(f"KMeans clusters:\n {kmeans.cluster_centers_}")
print(f"Points in cluster:\n {np.bincount(labels)}")

fig, ax = plt.subplots()
ax.scatter(X_clusters[:,0],X_clusters[:,1],c=labels)
ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color="black",marker="X",s=200)
plt.xlabel("Cluster X")
plt.ylabel("Cluster Y")
plt.title("Clusters Demo")
plt.savefig("outputs/kmeans_clusters.png")

plt.show()

# Linear Regression
# Medical costs dataset
np.random.seed(42)
num_patients = 100
age = np.random.randint(20,65,num_patients).astype(float)
smoker = np.random.randint(0,2,num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0,3000,num_patients)

# Linear Regression Q1
plt.scatter(age,cost,c=smoker,cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.show()

'''
There are two distinct groups visible and the age of the smoker 
shows that as the patient gets older, their healthcare cost will rise
'''

# Linear Regression Q2
X = age.reshape(-1,1)
y = cost
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print(f"\nLinear Regression Q2:\n")
print(f"X Train:\n {X_train}")
print(f"X Test:\n {X_test}")
print(f"Y Train:\n {y_train}")
print(f"Y test:\n {y_test}")

# Linear Regression Q3
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nLinear Regression Q3:\n")
print(f"Slope:",model.coef_[0])
print(f"Intercept:", model.intercept_)
print(f"RMSE:",np.sqrt(np.mean((y_pred - y_test) ** 2)))
print(f"R2:",model.score(X_test, y_test))

#The slope predicts how much the medical cost increases by age

# Linear Regression Q4
X_full = np.column_stack([age,smoker])
y = cost

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_full,y,test_size=0.2,random_state=42
)
model_full = LinearRegression()
model_full.fit(X_train2, y_train2)
y_pred2 = model_full.predict(X_test2)

print(f"\nLinear Regression Q4:\n")
print("age coefficient:", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
print(f"R2:{model_full.score(X_test2, y_test2)}\n")

'''
How does the smoker feature influence the predicted medical cost?
Adding the smoker flag helps R2 significantly by going from 0.07 to 0.77
The smoker coefficient represents the predicted cost of healthcare for all 100 patients
'''

# Linear Regression Q5

'''
The diagonal line represents perfect predictions where predicted cost
equals actual cost. Points above the line have actual costs higher than
predicted, while points below the line have actual costs lower than
predicted. Points closer to the diagonal represent more accurate
predictions.
'''

plt.scatter(y_pred2,y_test2,color="orange",cmap="coolwarm")
plt.plot( [0,30000],[0,30000], color="black")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual")
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()