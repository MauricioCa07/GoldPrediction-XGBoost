#import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

x = np.genfromtxt('XAU_15m_data.csv',skip_header=1, delimiter=',', usecols=(1,2,3,5))
y = np.genfromtxt('XAU_15m_data.csv',skip_header=1, delimiter=',', usecols=(4))


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123)

dtrain = xgb.DMatrix(data=x_train,label=y_train)  
dtest = xgb.DMatrix(data=x_test,label=y_test)

params = {
    'objective':'reg:squarederror',
    'max_depth':12,
    'learning_rate':0.01,
    'n_estimators':100,
    'alpha':10,
    'colsample_bytree': 1.0,
    'subsample': 1.0
}


#Best parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 12, 'subsample': 1.0}

model = xgb.XGBRegressor(**params)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)




mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")


#from sklearn.model_selection import GridSearchCV
#
#param_grid = {
#    'max_depth': [3, 6, 9,12,15,18,20,25,50,75,100],
#    'learning_rate': [0.01, 0.001, 0.0001,0.00001],
#    'subsample': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 1.0],
#    'colsample_bytree': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 1.0]
#}
#
#grid_search = GridSearchCV(
#    estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
#grid_search.fit(x_train, y_train)
#
#print("Best parameters:", grid_search.best_params_)