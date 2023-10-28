# smartsolve

![Screenshot from 2023-10-28 19-27-06](https://github.com/LoqmanSamani/smartsolve/assets/141053177/9d6cd280-4edd-4745-a9df-1ebc9e08a1cb)


### a mini-machine learning python package

**smartsolve** is a versatile machine learning package that empowers you to excel in data analysis and predictive modeling. With a rich array of machine learning models, it equips you to analyze diverse datasets, including text-based, numerical, and categorical data, and make precise predictions. The package is thoughtfully organized into three essential modules: *preprocessing*, *models*, and *evaluation*, each of which hosts a collection of classes with specialized functionalities.

## Table of Contents

- Installation
- Usage
- License

## Installation

Before installing and using `smartsolve`, please ensure that the following dependencies are installed on your system:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [scipy](https://www.scipy.org/)

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```  

Once the requirements are successfully installed, you can proceed to install `smartsolve`:

```python 
pip install smartsolve
```
For more information visit: [pypi.org](https://pypi.org/project/smartsolve/)

## Usage

**smartsolve** offers a comprehensive workflow for machine learning tasks. Here's how you can use it effectively:

1. **Data Preparation**: Utilize the `preprocessing` module to analyze and prepare your raw data. You can split your data into training, validation, and test sets, which can be used for various machine learning models available in the `models` module.

2. **Model Training**: Choose a machine learning model from the `models` module that suits your task. Train your model using the prepared data.

3. **Model Evaluation**: Once your model is trained, employ the `evaluation` module to evaluate its performance. You can assess various metrics like mean squared error (MSE), accuracy, precision, recall, F-score, and more.

Here's an example of how to use **smartsolve** to prepare data, train a model, and evaluate its performance:
```python
from smartsolve.preprocessing import AnalyseData, SplitData
from smartsolve.models import LinearRegression
from smartsolve.evaluation import Validation
import numpy as np
import pandas as pd
import seaborn as sns

# Load and analyze data from a text-based file

path = "example_data.csv"
example = AnalyseData(path)
data = example.load_data()

info = example.infos()
example.stats()
heatmap = example.heat_map(columns='data columns')


# A linear regression model as an example

# Prepare data for splitting
labels = list(data['labels_column'])
feats = ['feature_columns']
features = [[feature[i] for feature in feats] for i in range(len(labels))]

# Combine labels and features
input_data = [(label, feature) for label, feature in zip(labels, features)]

# Split the data into training and testing sets
split = SplitData(data=input_data, method='Random', train=0.8, test=0.2)
train_data, _, test_data = split.random()

# Train the linear regression model
model = LinearRegression(train_data=train_data, max_iter=400, threshold=1e-6)
model.train()
weights = model.coefficients
bias = model.bias

# Prepare test data
test_set = 'test_sample'

# Predict using the trained model
predicted = model.predict(data=test_set)

# Evaluate the model
obj = Validation()
mse = obj.mean_squared_error(actual='actual labels', predicted=predicted)

```
For more examples see this page :[examples](https://github.com/LoqmanSamani/smartsolve/tree/main/examples)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](https://github.com/LoqmanSamani/smartsolve/blob/main/smartsolve/LICENSE.txt) file for details.



