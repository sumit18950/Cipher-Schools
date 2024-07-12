# Linear Regression
from sklearn.model_selection import train_test_split.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Generating synthetic data
import numpy as np
x=np.random.rand(100,1)*10
y=2.5*x+np.random.randn(100,1)*2


#spliting data into training and testing sets...
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)  


#Training the model
model=LinearRegression()
model.fit(x_train,y_train)
#Predicting the test set results
y_pred=model.predict(x_test)
#Calculating the mean squared error (EVALUATION)
mse=mean_squared_error(y_test,y_pred)
print("mse->",mse)

#print(y_test)
