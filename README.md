# Ensemble-learning

## Project overview

This Ensemble learning project aims to demonstrate the power and effectiveness of ensemble learning techniques in the field of machine learning. Ensemble learning combines the predictions from multiple models to create a more accurate and reliable predictive model than any single model alone. This approach reduces the risk of overfitting and improves generalization to new data.

## Dataset
The Iris dataset, a classic dataset in machine learning, is used to implement and demonstrate these techniques. It contains 150 instances of iris flowers, each described by four features: sepal length, sepal width, petal length, and petal width. The goal is to classify the flowers into one of three species: Iris-setosa, Iris-versicolor, or Iris-virginica.

## Data Source: 
The dataset is sourced from the UCI Machine Learning Repository, originally published by Ronald A. Fisher in 1936.


![IRIS DATASET](https://github.com/adepel80/Ensemble-learning/assets/123180341/bc43ec6e-4bf2-4aaf-b9dd-fd0186b1e417)

``` from sklearn.datasets import load_iris
iris = load_iris()
X= iris.data
y= iris.target
```




## LIBRARIES
![ENSE LIBRAROES](https://github.com/adepel80/Ensemble-learning/assets/123180341/100604c3-2a38-47cf-95d3-7b08bb69bc5d)

```import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
plt.style.use ("dark_background")

import os
```

## TECHNIQUES IMPLEMENTED

### Bagging 
(Bootstrap Aggregating):

Involves training multiple versions of a model on different subsets of the training data and averaging their predictions.
Example: Random Forest, which combines multiple decision trees.
## Boosting:

Sequentially trains models, each focusing on correcting the errors of its predecessor.
Example: AdaBoost, Gradient Boosting Machines (GBM), XGBoost.

![ENSE ADA1](https://github.com/adepel80/Ensemble-learning/assets/123180341/8256d1bd-3d87-42a2-9063-b3e10edf6757)

## Stacking:

Combines multiple models using a meta-model, which is trained on the predictions of the base models.
Example: Using logistic regression or another algorithm as the meta-model.
![ENSE KNN](https://github.com/adepel80/Ensemble-learning/assets/123180341/6c5ad0a1-cfb3-49eb-a771-834f4d993e8e)

##  BAGGING ALGORITHM
![ENSEM BAGGING](https://github.com/adepel80/Ensemble-learning/assets/123180341/71c74d4b-9566-485a-9579-aff4074bc2bc)


## Project Structure

Data: Raw and processed data files.
Notebooks: Jupyter notebooks demonstrating data preprocessing, exploratory data analysis, model training, and evaluation.
Scripts: Python scripts for preprocessing data, training models, and evaluating model performance.
Reports: Performance metrics and visualizations.

## Key Findings
Ensemble methods like Random Forest, Gradient Boosting, and Stacking have shown significant improvement in classification accuracy and model robustness compared to individual models. Detailed performance metrics and visualizations illustrate these improvementS.


