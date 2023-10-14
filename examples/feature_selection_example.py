from learnflow.preprocessing import SelectFeature



# Example 1: Correlation Selection
example1 = SelectFeature(csv_file='example_data.csv', label='a')
# Get features with a correlation above 0.7 with the label 'a'
features, data = example1.correlation(threshold=0.7)




# Example 2: Mutual Information
example2 = SelectFeature(csv_file='example_data.csv')
# Calculate mutual information for each feature
mutual_info = example1.mutual_infos()




# Example 3: Lasso Regularization
example_coefficients = [0.2, -0.3, 0.5, 0.1, -0.7, 0.4, 0.6, -0.2, 0.8, -0.9, 0.2, -0.3, 0.5, 0.1, -0.7, 0.4, 0.6, -0.2, 0.8, -0.9]
example3 = SelectFeature()
# Apply Lasso regularization to example_coefficients
new_coefficients = example3.lasso(example_coefficients, lam=0.01, learning_rate=0.01, threshold=1e-4)




# Example 4: Select Best Features (Numerical)
example4 = SelectFeature(csv_file='example_data.csv', label='a')
# Select the 2 best numerical features based on their importance
best_features = example4.best_features(k=2, data_type='numerical')





# Example 5: Select Best Features (Categorical)
example5 = SelectFeature(csv_file='categorical_data.csv', label='CategoryA')
# Select the 2 best categorical features based on their importance
best_features1 = example5.best_features(k=2, data_type='categorical')




# Example 6: Variance Threshold
example6 = SelectFeature(csv_file='sample_data.csv', label='Label')
threshold = 0.1  # Set your desired threshold value
# Remove features with variance below the specified threshold
filtered_data = example6.variance_threshold(threshold=threshold)




"""
Output:


--------------------------------------------------
**Correlation**
--------------------------------------------------
['a', 'b']
   a   b
0  1   2
1  2   4
2  3   6
3  4   8
4  5  10
5  6  12

--------------------------------------------------
**Mutual_information**
--------------------------------------------------

{'b': 2.584962500721156, 'c': 1.5849625007211556, 'd': 2.584962500721156}


--------------------------------------------------
**Lasso Regularization**
--------------------------------------------------

[0.1312000000000076, -0.23120000000000757, 0.4312000000000076, 0.03119999999999804,
 -0.6312000000000075, 0.3312000000000076, 0.5312000000000076, -0.1312000000000076,
  0.7312000000000076, -0.8312000000000076, 0.1312000000000076, -0.23120000000000757,
   0.4312000000000076, 0.03119999999999804, -0.6312000000000075, 0.3312000000000076,
    0.5312000000000076, -0.1312000000000076, 0.7312000000000076, -0.8312000000000076]

--------------------------------------------------
**Best k Features (numerical data)**
--------------------------------------------------

['d', 'b']

--------------------------------------------------
**Best k Features (categorical data)**
--------------------------------------------------

['CategoryB', 'CategoryC']

--------------------------------------------------
**Variance Threshold**
--------------------------------------------------

   FeatureA  FeatureB  Label
0         1         1      0
1         2         2      1
2         3         1      0
3         4         2      1
4         5         1      0

"""