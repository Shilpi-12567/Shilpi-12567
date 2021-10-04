import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier 
import joblib
#reading data 
bmi_data=pd.read_csv(r"Bmi_male_female.csv")
#print(bmi_data)
#print(bmi_data.keys())
X_feature=bmi_data.iloc[:,0:3]
Y_target=bmi_data.iloc[:,3]
X_feature["Gender"]=X_feature["Gender"].map({"Male":0,"Female":1})
#print(X_feature)
model_trainer=KNeighborsClassifier(n_neighbors=5)
#print(model_trainer)
model_learer=model_trainer.fit(X_feature,Y_target)
#model_learer.predict([[0,180,29]])
#print(model_learer.predict([[0,180,29]]))
index_target=pd.Series(["Extremely Weak","Weak" ,"Normal" ,"Overweight","Obesity" ,"Extreme Obesity"])
prediction=model_learer.predict([[0,180,29]])
result=index_target[prediction]
result=list(result.values)
result=str(result)
print(type(result))
print(result)
#saving the model
filename = 'bmi.pkl'
joblib.dump(model_learer, filename)