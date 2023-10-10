# Import necessary libraries and modules
from split import SplitData
import pandas as pd
import random
from datetime import datetime, timedelta



# Define the path to the CSV data file
path = ("...")


# Read the data from the CSV file into a DataFrame
data = pd.read_csv(path)


# Extract column names from the DataFrame
columns = data.columns


# Extract labels from the 'Rank' column
labels = data['Rank']


# Select features for training
features = data[['Categories', 'Suscribers', 'Country', 'Visits', 'Comments']]


# Prepare features as a list of lists
lst_features = []
for index, row in features.iterrows():
    point = list(row)
    lst_features.append(point)



# Create data in the required format for SplitData class
data = list(zip(labels, lst_features))




# Initialize a SplitData instance for random splitting
random_model = SplitData(data=data, method='Random', train=0.8, validation=0.1, test=0.20)

# Perform random split
r_train, r_validation, r_test = random_model.random()



# Split data into data points for stratified splitting
data_points = [data[:200], data[200:450], data[450:920], data[920:]]

# Initialize a SplitData instance for stratified splitting
stratified_model = SplitData(data=data_points, method='Stratified', train=0.8, validation=0.15, test=0.20)

# Perform stratified split
s_train, s_test, s_validation = stratified_model.stratified()






# Generate a list of random dates for time series splitting
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)
date_list = []

for _ in range(1000):
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    date_list.append(random_date)

# Initialize a SplitData instance for time series splitting
time_series_model = SplitData(data=data, method='TimeSeries', train=0.8, validation=None, test=0.20, date=date_list)

# Perform time series split
t_train, t_validation, t_test = time_series_model.time_series()






# Initialize a SplitData instance for K-fold cross-validation
cross_validation_model = SplitData(data=data, method='KFold', train=80, validation=0.1, test=0.20, num_folds=20)

# Perform K-fold cross-validation
c_train, c_validation, c_test = cross_validation_model.cross_validation()







# Print the shapes of the resulting splits
print("Random Split Shapes:")
print(len(r_train), len(r_validation), len(r_test))
print("\nStratified Split Shapes:")
print(len(s_train), len(s_test), len(s_validation))
print("\nTime Series Split Shapes:")
print(len(t_train), len(t_validation), len(t_test))
print("\nK-Fold Cross-Validation Shapes:")
print([len(f) for f in c_train])
print([len(f) for f in c_validation])
print([len(f) for f in c_test])

"""
Output:
Random Split Shapes:
720 80 200

Stratified Split Shapes:
651 149 200

Time Series Split Shapes:
800 0 200

K-Fold Cross-Validation Shapes:
[850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850]
[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

"""



