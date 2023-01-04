import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df_2020 = pd.read_csv("Passing_2020.csv")
df_2021 = pd.read_csv("Passing_2021.csv")
#Calculating touchdown, sack, and interception percentage for each player
df_2020["TD%"] = (df_2020['TD'] * 100) / df_2020['ATT']
df_2021["TD%"] = (df_2021['TD'] * 100) / df_2021['ATT']
df_2020["INT%"] = (df_2020['INT'] * 100) / df_2020['ATT']
df_2021["INT%"] = (df_2021['INT'] * 100) / df_2021['ATT']
df_2020["SACK%"] = (df_2020['SACK'] * 100) / df_2020['ATT']
df_2021["SACK%"] = (df_2021['SACK'] * 100) / df_2021['ATT']
#Normalizing data points
df_2020["TD%"] = df_2020["TD%"]  / df_2020["TD%"].abs().max()
df_2020["INT%"] = df_2020["INT%"]  / df_2020["INT%"].abs().max()
df_2020["ATT"] = df_2020["ATT"]  / df_2020["ATT"].abs().max()
df_2020["CMP"] = df_2020["CMP"]  / df_2020["CMP"].abs().max()
df_2020["TD"] = df_2020["TD"]  / df_2020["TD"].abs().max()
df_2020["INT"] = df_2020["INT"]  / df_2020["INT"].abs().max()
df_2020["YDS"] = df_2020["YDS"]  / df_2020["YDS"].abs().max()
df_2020["SACK"] = df_2020["SACK"]  / df_2020["SACK"].abs().max()
df_2020["SACK%"] = df_2020["SACK%"]  / df_2020["SACK%"].abs().max()
df_2020["AVG"] = df_2020["AVG"]  / df_2020["AVG"].abs().max()
df_2021["TD%"] = df_2021["TD%"]  / df_2021["TD%"].abs().max()
df_2021["INT%"] = df_2021["INT%"]  / df_2021["INT%"].abs().max()
df_2021["ATT"] = df_2021["ATT"]  / df_2021["ATT"].abs().max()
df_2021["CMP"] = df_2021["CMP"]  / df_2021["CMP"].abs().max()
df_2021["TD"] = df_2021["TD"]  / df_2021["TD"].abs().max()
df_2021["INT"] = df_2021["INT"]  / df_2021["INT"].abs().max()
df_2021["YDS"] = df_2021["YDS"]  / df_2021["YDS"].abs().max()
df_2021["SACK"] = df_2021["SACK"]  / df_2021["SACK"].abs().max()
df_2021["SACK%"] = df_2021["SACK%"]  / df_2021["SACK%"].abs().max()
df_2021["AVG"] = df_2021["AVG"]  / df_2021["AVG"].abs().max()
#Using touchdown and interception percentage to predict QBR
X_1 = df_2021[["TD%", "INT%"]]
y_1 = df_2021["QBR"]
model1 = LinearRegression()
model1 = model1.fit(df_2020[['TD%',"INT%"]], df_2020["QBR"])
df_1 = df_2021.copy()
df_1["Predicted QBR"] = model1.predict(df_2021[["TD%", "INT%"]])
df_1["Residual"] = df_1["QBR"] - df_1["Predicted QBR"]
RMSE_1 = mean_squared_error(df_1["QBR"], df_1["Predicted QBR"], squared = False)
print("The accuracy for predicting QBR from touchdown percentage and interception percentage is: " + str(round(model1.score(X_1, y_1) * 100, 3)) + "%")
#Using yards per attempt and completion percentage to predict QBR
X_2 = df_2021[["AVG", "CMP%"]]
y_2 = df_2021["QBR"]
model2 = LinearRegression()
model2 = model2.fit(df_2020[['AVG','CMP%']], df_2020["QBR"])
df_2 = df_2021.copy()
df_2["Predicted QBR"] = model2.predict(df_2021[["AVG", "CMP%"]])
df_2["Residual"] = df_2["QBR"] - df_2["Predicted QBR"]
RMSE_2 = mean_squared_error(df_2["QBR"], df_2["Predicted QBR"], squared = False)
print("The accuracy for predicting QBR from YPA and completion percentage is: " + str(round(model2.score(X_2, y_2) * 100, 3)) + "%")
#Using interception and completion percentage to predict QBR
X_3 = df_2021[["INT%", "CMP%"]]
y_3 = df_2021["QBR"]
model3 = LinearRegression()
model3 = model3.fit(df_2020[['INT%','CMP%']], df_2020["QBR"])
df_3 = df_2021.copy()
df_3["Predicted QBR"] = model3.predict(df_2021[["INT%", "CMP%"]])
df_3["Residual"] = df_3["QBR"] - df_3["Predicted QBR"]
RMSE_3 = mean_squared_error(df_3["QBR"], df_3["Predicted QBR"], squared = False)
print("The accuracy for predicting QBR from interception percentage and completion percentage is: " + str(round(model3.score(X_3, y_3) * 100, 3)) + "%")
#Using touchdown and completion percentage to predict QBR
X_4 = df_2021[["TD%", "CMP%"]]
y_4 = df_2021["QBR"]
model4 = LinearRegression()
model4 = model4.fit(df_2020[['TD%','CMP%']], df_2020["QBR"])
df_4= df_2021.copy()
df_4["Predicted QBR"] = model4.predict(df_2021[["TD%", "CMP%"]])
df_4["Residual"] = df_4["QBR"] - df_4["Predicted QBR"]
RMSE_4 = mean_squared_error(df_4["QBR"], df_4["Predicted QBR"], squared = False)
print("The accuracy for predicting QBR from touchdown percentage and completion percentage is: " + str(round(model4.score(X_4, y_4) * 100, 3)) + "%")
#Using yards per attempt and interception percentage to predict QBR
X_5 = df_2021[["AVG", "INT%"]]
y_5 = df_2021["QBR"]
model5 = LinearRegression()
model5 = model5.fit(df_2020[['AVG','INT%']], df_2020["QBR"])
df_5 = df_2021.copy()
df_5["Predicted QBR"] = model5.predict(df_2021[["AVG", "INT%"]])
df_5["Residual"] = df_5["QBR"] - df_5["Predicted QBR"]
RMSE_5 = mean_squared_error(df_5["QBR"], df_5["Predicted QBR"], squared = False)
print("The accuracy for predicting QBR from YPA and interception percentage is: " + str(round(model5.score(X_5, y_5) * 100, 3)) + "%")
#Using sack and interception percentage to predict QBR
X_6 = df_2021[["SACK%", "INT%"]]
y_6 = df_2021["QBR"]
model6 = LinearRegression()
model6 = model6.fit(df_2020[['SACK%','INT%']], df_2020["QBR"])
df_6 = df_2021.copy()
df_6["Predicted QBR"] = model6.predict(df_2021[["SACK%", "INT%"]])
df_6["Residual"] = df_6["QBR"] - df_6["Predicted QBR"]
RMSE_6 = mean_squared_error(df_6["QBR"], df_6["Predicted QBR"], squared = False)
print("The accuracy for predicting QBR from sack percentage and interception percentage is: " + str(round(model6.score(X_6, y_6) * 100, 3)) + "%")
print("The RMSE for model 1 is: " + str(RMSE_1))
print("The RMSE for model 2 is: " + str(RMSE_2))
print("The RMSE for model 3 is: " + str(RMSE_3))
print("The RMSE for model 4 is: " + str(RMSE_4))
print("The RMSE for model 5 is: " + str(RMSE_5))
print("The RMSE for model 6 is: " + str(RMSE_6))