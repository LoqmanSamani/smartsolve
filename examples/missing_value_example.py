import pandas as pd
import numpy as np
from learnflow.preprocessing import MissingValue





# Load data from a CSV file
# data: Customer Satisfaction in Airline, https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline
path = "Invistico_Airline.csv"



example1 = MissingValue(path)
cols = example1.load_data()
print('Open a CSV file:')
print(cols)





# Create a DataFrame with missing values
dataframe = pd.DataFrame({'Name': ['ALI', 'Amin', 'Saman', 'hashem', np.nan],
                          'age': [21, np.nan, np.nan, 65, 12],
                          'single': ['no', 'no', np.nan, 'yes', np.nan]
                          })



# Example 2: Replace missing values with 'replaced!'
example2 = MissingValue(data=dataframe, replace='Value', rep_value='replaced!')
modified = example2.numerical()
print('Original DataFrame:')
print(dataframe)
print('Modified:')
print(modified)






# Create another DataFrame with missing values
dataframe1 = pd.DataFrame({'Name': ['ALI', 'Amin', 'Saman', 'hashem', np.nan],
                          'age': [21, np.nan, np.nan, 65, 12],
                          'single': ['no', 'no', np.nan, 'yes', np.nan]
                           })





# Example 3: Delete rows containing any missing values and replace with 'Yes!!!'
example3 = MissingValue(data=dataframe1, replace='Del')
modified1 = example3.qualitative()
print('Modified 1:')
print(modified1)





"""
Output:

Open a CSV file:
        satisfaction  ... Arrival Delay in Minutes
0          satisfied  ...                      0.0
1          satisfied  ...                    305.0
2          satisfied  ...                      0.0
3          satisfied  ...                      0.0
4          satisfied  ...                      0.0
...              ...  ...                      ...
129875     satisfied  ...                      0.0
129876  dissatisfied  ...                    172.0
129877  dissatisfied  ...                    163.0
129878  dissatisfied  ...                    205.0
129879  dissatisfied  ...                    186.0

[129880 rows x 22 columns]


Original DataFrame:

     Name   age single
0     ALI  21.0     no
1    Amin   NaN     no
2   Saman   NaN    NaN
3  hashem  65.0    yes
4     NaN  12.0    NaN


Modified:

        Name        age     single
0        ALI       21.0         no
1       Amin  replaced!         no
2      Saman  replaced!  replaced!
3     hashem       65.0        yes
4  replaced!       12.0  replaced!


Modified 1:

     Name   age single
0     ALI  21.0     no
3  hashem  65.0    yes

"""


