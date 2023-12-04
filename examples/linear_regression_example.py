# Import necessary modules and classes

from smartsolve.models import LinearRegression
from smartsolve.preprocessing import SplitData, AnalyseData
from machine_learning.linear_algebra import intro_numpy as np

# Load and analyze the dataset
path = "/home/sam/python_projects/data_sets/regression/Real estate.csv"

data = AnalyseData(data=path)
load = data.load_data()

info = data.infos()
print(info)

stats = data.stats()
print(stats)

heat_map = data.heat_map()


# Select relevant columns for the regression task
rel_cols = load[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station',
                  'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]



# Prepare data for linear regression
labels = list(load['Y house price of unit area'])
feats = [list(rel_cols[column]) for column in rel_cols]
features = [[feature[i] for feature in feats] for i in range(len(labels))]
input_data = [(label, feature) for label, feature in zip(labels, features)]



# Split the data into training and testing sets
split = SplitData(data=input_data, method='Random', train=0.8, test=0.2)
train_data, _, test_data = split.random()




# Example for Ridge Regression
model_rr = LinearRegression(train_data=train_data, algorithm='RR', max_iter=400, threshold=1e-6, alpha=0.01)
model_rr.train()
weights_rr = model_rr.coefficients
bias_rr = model_rr.bias




# Prepare test features
test_features = np.array([item[1] for item in test_data])




# Predict with Ridge Regression
predicted_rr = model_rr.predict(features=test_features, coefficients=weights_rr, bias=bias_rr)
print("Ridge Regression Predictions:")
print(predicted_rr)




# Example for Gradient Descent (GD)
model_gd = LinearRegression(train_data=train_data, algorithm='GD', max_iter=400, learning_rate=0.01)
model_gd.train()
weights_gd = model_gd.coefficients
bias_gd = model_gd.bias





# Predict with Gradient Descent
predicted_gd = model_gd.predict(features=test_features, coefficients=weights_gd, bias=bias_gd)
print("Gradient Descent Predictions:")
print(predicted_gd)




# Example for Stochastic Gradient Descent (SGD)
model_sgd = LinearRegression(train_data=train_data, algorithm='SGD', max_iter=400, learning_rate=0.01)
model_sgd.train()
weights_sgd = model_sgd.coefficients
bias_sgd = model_sgd.bias




# Predict with Stochastic Gradient Descent
predicted_sgd = model_sgd.predict(features=test_features, coefficients=weights_sgd, bias=bias_sgd)
print("Stochastic Gradient Descent Predictions:")
print(predicted_sgd)




