import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/linux/Documents/Test_App/csv_file.csv')
df = df.drop(['Name','EU_Sales','NA_Sales','JP_Sales','Other_Sales','Year'], axis = 1)
df['y'] = pd.qcut(df['Global_Sales'], q = [0, 0.25, 0.5, 0.75, 1], labels = [1,2,3,4])

# Dichotomisation des variables catégorielles
publisher = pd.get_dummies(df['Publisher'], prefix = 'publisher')
platform = pd.get_dummies(df['Platform'], prefix = 'platform')
genre = pd.get_dummies(df['Genre'], prefix = 'genre')

# Ajout des colonnes dichotomisée au DataFrame
df = df.join(publisher)
df = df.join(platform)
df = df.join(genre)

# Homogénéisation du barème des notes (tout est ramené sur 10)
df['Test_MC'] = df['Test_MC'] / 10
df['Test_JV'] = df['Test_JV'] / 2
df['Players_JV'] = df['Players_JV'] / 2


# Suppression des colonnes inutiles
df = df.drop(['Publisher','Platform','Genre', 'Global_Sales', 'NA_Sales','EU_Sales','JP_Sales','Other_Sales','Year'], axis = 1)

# Feats & Target
X = df.drop('y', axis = 1)
Y = df['y']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Scaler
sc = MinMaxScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

##### ARBRE DE DECISION CLASSIFICATION

model_classification = DecisionTreeClassifier(random_state = 0)
model_classification.fit(X_train_sc, y_train)

# Entraîner le pipeline modèle
model_classification.fit(X_train, y_train)

# Save the model as a pickle in a file
joblib.dump(model_classification, '/home/linux/Documents/Test_App/models/model_classification.pkl')
X.to_csv('X_file.csv', index = False)
Y.to_csv('Y_file.csv', index = False)