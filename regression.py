import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer    
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/linux/Documents/Test_App/csv_file.csv')
df = df.drop(['Name','EU_Sales','NA_Sales','JP_Sales','Other_Sales','Year'], axis = 1)

X, Y = df.drop('Global_Sales', axis =1), df['Global_Sales']

# Séparation des variables numériques et catégorielles
num_vars = X.select_dtypes(exclude=['object']).columns
cat_vars = X.drop(num_vars, axis = 1).columns

# Transformateur numérique
numeric_transformer = make_pipeline(
    (SimpleImputer(strategy = 'median')), 
    (MinMaxScaler())
)

# Transformateur catégorielle
categorical_transformer = make_pipeline(
    (SimpleImputer(strategy ='most_frequent')),
    (OneHotEncoder(handle_unknown="ignore"))
)

# Combinaison des transformateur : preprocessor
preprocessor = ColumnTransformer(
    transformers = [('num', numeric_transformer, num_vars),('cat', categorical_transformer, cat_vars)]
)

# Pipeline finale
model_regression = Pipeline(
    steps = [('preprocessing',preprocessor),('classification',DecisionTreeRegressor(random_state = 0))]
)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)

# Entraîner le pipeline modèle
model_regression.fit(X_train, y_train)

# Save the model as a pickle in a file
joblib.dump(model_regression, '/home/linux/Documents/Test_App/models/model_classification.pkl')
  
"""# Load the model from the file
model_classification_from_joblib = joblib.load('/home/linux/Documents/Test_App/models/model_classification.pkl')
  
# Use the loaded model to make predictions
model_classification_from_joblib.predict(X_test)"""
