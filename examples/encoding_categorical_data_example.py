import pandas as pd
from smartsolve.preprocessing import CategoricalData



# Sample DataFrame
data = pd.DataFrame({
    'Category1': ['A', 'B', 'C', 'A', 'B', 'G', 'A', 'B', 'G'],
    'Category2': ['X', 'Y', 'X', 'Y', 'Z', 'Z', 'Z', 'Z', 'X'],
    'Category3': ['A', 'X', 'X', 'Y', 'Z', 'B', 'B', 'Q', 'Q']
})



# Labels for encoding
labels = ['A', 'B', 'G', 'C', 'X', 'Q', 'Z', 'Y']

# List of desired encoding values
values = [10, 20, 30, 40, 50, 60, 70, 80]

# Create an example instance of CategoricalData
example = CategoricalData()


# Label Encoding
label_encoding = example.l_encoding(data=data, labels=labels, nums=values)
print("Label Encoding:")
print(label_encoding)

# One-Hot Encoding
one_hot_encoding = example.onehot_encoding(data=data, yes=122, no=1.4)
print("\nOne-Hot Encoding:")
print(one_hot_encoding)
print("Columns in One-Hot Encoding:")
print(one_hot_encoding.columns)

# Binary Encoding
binary_encoding = example.bin_encoding(data=data, labels=labels, nums=None, index=10)
print("\nBinary Encoding:")
print(binary_encoding)

# Count Encoding
count_encoding = example.count_encoding(data=data)
print("\nCount Encoding:")
print(count_encoding)

# Mean Encoding (Target Encoding)
mean_encoding = example.mean_encoding(data=data)
print("\nMean Encoding (Target Encoding):")
print(mean_encoding)

# Frequency Encoding
frequency_encoding = example.freq_encoding(data=data, r=13)
print("\nFrequency Encoding:")
print(frequency_encoding)


"""
---------------
Label Encoding:
---------------
   Category1  Category2  Category3
0         10         50         10
1         20         80         50
2         40         50         50
3         10         80         80
4         20         70         70
5         30         70         20
6         10         70         20
7         20         70         60
8         30         50         60
-------------------------------------------------------------------------------------
One-Hot Encoding:
-----------------
   Category1_C  Category1_B  Category1_G  ...  Category3_A  Category3_Z  Category3_Q
0          1.4          1.4          1.4  ...        122.0          1.4          1.4
1          1.4        122.0          1.4  ...          1.4          1.4          1.4
2        122.0          1.4          1.4  ...          1.4          1.4          1.4
3          1.4          1.4          1.4  ...          1.4          1.4          1.4
4          1.4        122.0          1.4  ...          1.4        122.0          1.4
5          1.4          1.4        122.0  ...          1.4          1.4          1.4
6          1.4          1.4          1.4  ...          1.4          1.4          1.4
7          1.4        122.0          1.4  ...          1.4          1.4        122.0
8          1.4          1.4        122.0  ...          1.4          1.4        122.0

[9 rows x 13 columns]
Columns in One-Hot Encoding:
Index(['Category1_C', 'Category1_B', 'Category1_G', 'Category1_A',
       'Category2_Y', 'Category2_X', 'Category2_Z', 'Category3_X',
       'Category3_Y', 'Category3_B', 'Category3_A', 'Category3_Z',
       'Category3_Q'],
      dtype='object')
---------------------------------------------------------------------------------------
Binary Encoding:
----------------
  Category1 Category2 Category3
0      1010      1110      1010
1      1011     10001      1110
2      1101      1110      1110
3      1010     10001     10001
4      1011     10000     10000
5      1100     10000      1011
6      1010     10000      1011
7      1011     10000      1111
8      1100      1110      1111
-------------------------------
Count Encoding:
---------------
{'Category1': [('C', 1), ('B', 3), ('G', 2), ('A', 3)], 'Category2': [('Y', 2), ('X', 3), ('Z', 4)], 'Category3':
 [('X', 2), ('Y', 1), ('B', 2), ('A', 1), ('Z', 1), ('Q', 2)]}
-----------------------------------------------------------------------------------------------------------------
Mean Encoding (Target Encoding):
--------------------------------
{'Category1': {'C': 2.0, 'B': 4.0, 'G': 5.0, 'A': 3.0}, 'Category2': {'Y': 2.0, 'X': 1.0, 'Z': 5.5}, 'Category3':
 {'X': 1.5, 'Y': 3.0, 'B': 5.5, 'A': 0.0, 'Z': 4.0, 'Q': 7.0}}
-----------------------------------------------------------------------------------------------------------------
Frequency Encoding:
------------------
{'Category1': [('C', 0, 13), ('B', 0, 13), ('G', 0, 13), ('A', 0, 13)], 'Category2': [('Y', 0, 13), ('X', 0, 13),
 ('Z', 0, 13)], 'Category3': [('X', 0, 13), ('Y', 0, 13), ('B', 0, 13), ('A', 0, 13), ('Z', 0, 13), ('Q', 0, 13)]}

"""