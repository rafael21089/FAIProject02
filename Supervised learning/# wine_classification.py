# wine_classification.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
import io
import graphviz

#Carregar dados e preparação/analise dos mesmos
wine_data = pd.read_csv('D:/_MIAA/FIA/_Proj02_M23_v11dez/winequality-red.csv')
wine_data.head(10)
#Dataset size
wine_data.shape
#Statistics
wine_data.describe()
#Data details
wine_data.info()
#Search for NULL values
wine_data.isnull().sum()

#Visualização dos dados

#Visualização Votação (qualidade)
sns.catplot(x='quality', data=wine_data, kind='count', color='green')
plt.xlabel('Qualidade')
plt.ylabel('Nº Votos')
plt.title('Votação Qualidade')
plt.show()

#Analise como "fixed acidity" influencia a  qualidade.

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine_data, color='green', errorbar=None)
#sns.barplot(x = 'quality', y = 'fixed acidity', data = wine_data, color='green', ci=None)
plt.xlabel('Qualidade')

#Aqui podemos ver que "volatile acidity" desce quando a qualidade melhora

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine_data, color='green', errorbar=None)
plt.xlabel('Qualidade')

#Quanto maior for a qualidade maior é a quantidade de "Sulphates"
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = wine_data, color='green', errorbar=None)
plt.xlabel('Qualidade')

#Correlation map
correlation = wine_data.corr()
#Visualização Correlação
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, annot=True, cmap='Greens')

#Preparação dos dados para usar os modelos de ML

#Remover a coluna "quality" e criar um novo dataset
X = wine_data.drop('quality', axis=1)
X.head()

#Definir qualide como boa ou má >6 = boa e <6 = má
y = wine_data['quality'].apply(lambda y_value: 1 if y_value>=6 else 0)
y

##RANDOM FOREST CLASSIFIER
#
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y.shape, y_train.shape, y_test.shape)

# Supervised Learning - Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
accuracy_rfc = accuracy_score(y_test, rf_predictions)
report = classification_report(y_test, rf_predictions, zero_division=0)

print("\nRandom Forest Accuracy:", accuracy_rfc*100)
print("Classification Report:\n", report)
from sklearn import metrics
print("\n Confusion Matrix:")
print(metrics.confusion_matrix(y_test, rf_model.predict(X_test)))

#Testes Random Forest Classifier

input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

# Assuming 'feature_names' is a list of your feature names
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Create a DataFrame from your input data
input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)

prediction = rf_model.predict(input_data_df)
print(prediction)

if (prediction[0]==1):
  print('Vinho de Boa Qualidade')
else:
  print('Vinho de Má Qualidade')

##SVM - Stochastic Gradient Decent Classifier
#

sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)

print(y.shape, y_train.shape, y_test.shape)

# Evaluate the model
accuracy_sgd = accuracy_score(y_test, pred_sgd)

print("\nStochastic Gradient Decent Classifier:", accuracy_sgd*100)
print(classification_report(y_test, pred_sgd, zero_division=1))
from sklearn import metrics
print("\n Confusion Matrix:")
print(metrics.confusion_matrix(y_test, rf_model.predict(X_test)))

#Testes Stochastic Gradient Decent Classifier

#input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5) #Má Qualidade
input_data = (7.3,0.65,0.00,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)

# Assuming 'feature_names' is a list of your feature names
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Create a DataFrame from your input data
input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)

prediction = rf_model.predict(input_data_df)
print(prediction)

if (prediction[0]==1):
  print('Vinho de Boa Qualidade')
else:
  print('Vinho de Má Qualidade')


## Support Vector Classifier
#
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

print(y.shape, y_train.shape, y_test.shape)

# Evaluate the model
accuracy_svc = accuracy_score(y_test, pred_svc)

print("\nSupport Vector Classifier:", accuracy_svc*100)
print(classification_report(y_test, pred_svc, zero_division=1))
from sklearn import metrics
print("\n Confusion Matrix:")
print(metrics.confusion_matrix(y_test, rf_model.predict(X_test)))

#Testes Support Vector Classifier

#input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5) #Má Qualidade
input_data = (7.3,0.66,0.00,1.1,0.065,15.0,21.0,0.9936,3.49,0.47,10.1)

# Assuming 'feature_names' is a list of your feature names
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Create a DataFrame from your input data
input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)

prediction = rf_model.predict(input_data_df)
print(prediction)

if (prediction[0]==1):
  print('Vinho de Boa Qualidade')
else:
  print('Vinho de Má Qualidade')


## KNN - K Nearest Neighbors Classifier
#KNN  Algorithm

modell = KNeighborsClassifier(n_neighbors = 5)
modell.fit(X_train, y_train)

pred_knn = modell.predict(X_test)

# Evaluate the model
accuracy_Knn = accuracy_score(y_test, pred_knn)

print("\nKNN Algorithm:", accuracy_Knn*100)
print(classification_report(y_test, pred_knn, zero_division=1))

from sklearn import metrics
print("\n Confusion Matrix:")
print(metrics.confusion_matrix(y_test, rf_model.predict(X_test)))

#Testes KNN Algorithm

#input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5) #Má Qualidade
input_data = (5.3,0.46,0.00,1.2,0.065,16.0,21.0,0.9836,3.59,0.47,10.1)

# Assuming 'feature_names' is a list of your feature names
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Create a DataFrame from your input data
input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)

prediction = rf_model.predict(input_data_df)
print(prediction)

if (prediction[0]==1):
  print('Vinho de Boa Qualidade')
else:
  print('Vinho de Má Qualidade')


#Comparação dos Algoritmos

models = {
    "models": ['Random Forest Classifier','Stochastic Gradient Decent Classifier','Support Vector Classifier','KNN Algorithm'],
    "score": [accuracy_rfc, accuracy_sgd, accuracy_svc, accuracy_Knn]
}
models

#Visualizar Comparação dos Algoritmos

colors = ['green', 'blue', 'red', 'yellow']
sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
ax = sns.barplot(x=models['models'], y=models['score'], hue=models['models'], palette=colors, legend=False)
plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Model Selection")

plt.show()

##Como podemos ver o melhor modelo é o Random Forest Classifier
#Analisando mais alguns detalhes dos dados usando Random Forest Classifier
# Feature Importances in wine quality dataset

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
sorted_feature_importances = feature_importances.sort_values(ascending=False)

# Print and Visualize Feature Importances
print("\nFeature Importances:")
print(sorted_feature_importances)

# Plot Feature Importances
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_feature_importances.values, y=sorted_feature_importances.index, hue=sorted_feature_importances.index, palette="viridis", legend=False)
plt.title('Random Forest - Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

##Visualizar a árvore de decisão
#
# Visualize one of the decision trees in the forest
estimator = rf_model.estimators_[0]
dot_data = export_graphviz(estimator, out_file=None, feature_names=X.columns, class_names=[str(i) for i in range(3)],
                           rounded=True, filled=True)
graph = graphviz.Source(dot_data)

# Save the decision tree as an image file (optional)
graph.render(filename='tree', format='png', cleanup=True)

# Display the decision tree using graphviz
graph.view()
