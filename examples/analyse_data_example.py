from smartsolve.preprocessing import AnalyseData


# Load data from a CSV file
# data: Customer Satisfaction in Airline, https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline
path = "Invistico_Airline.csv"

example = AnalyseData(path)

data = example.load_data()

info = example.infos()

example.stats()

columns = ["Arrival Delay in Minutes", "Departure Delay in Minutes", "Online boarding"]

heatmap = example.heat_map(columns=columns)



"""
Output:

--------------------------------------------------
**Load Data**
--------------------------------------------------
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
--------------------------------------------------
**Infos**
--------------------------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 129880 entries, 0 to 129879
Data columns (total 22 columns):
 #   Column                             Non-Null Count   Dtype  
---  ------                             --------------   -----  
 0   satisfaction                       129880 non-null  object 
 1   Customer Type                      129880 non-null  object 
 2   Age                                129880 non-null  int64  
 3   Type of Travel                     129880 non-null  object 
 4   Class                              129880 non-null  object 
 5   Flight Distance                    129880 non-null  int64  
 6   Seat comfort                       129880 non-null  int64  
 7   Departure/Arrival time convenient  129880 non-null  int64  
 8   Food and drink                     129880 non-null  int64  
 9   Gate location                      129880 non-null  int64  
 10  Inflight wifi service              129880 non-null  int64  
 11  Inflight entertainment             129880 non-null  int64  
 12  Online support                     129880 non-null  int64  
 13  Ease of Online booking             129880 non-null  int64  
 14  On-board service                   129880 non-null  int64  
 15  Leg room service                   129880 non-null  int64  
 16  Baggage handling                   129880 non-null  int64  
 17  Checkin service                    129880 non-null  int64  
 18  Cleanliness                        129880 non-null  int64  
 19  Online boarding                    129880 non-null  int64  
 20  Departure Delay in Minutes         129880 non-null  int64  
 21  Arrival Delay in Minutes           129487 non-null  float64
dtypes: float64(1), int64(17), object(4)
memory usage: 21.8+ MB
None
--------------------------------------------------
**Statistics**
--------------------------------------------------
The satisfaction is a categorical column.
satisfaction
satisfied       71087
dissatisfied    58793
Name: count, dtype: int64
The Customer Type is a categorical column.
Customer Type
Loyal Customer       106100
disloyal Customer     23780
Name: count, dtype: int64
The Age is a numerical column.
count    129880.000000
mean         39.427957
std          15.119360
min           7.000000
25%          27.000000
50%          40.000000
75%          51.000000
max          85.000000
Name: Age, dtype: float64
The Type of Travel is a categorical column.
Type of Travel
Business travel    89693
Personal Travel    40187
Name: count, dtype: int64
The Class is a categorical column.
Class
Business    62160
Eco         58309
Eco Plus     9411
Name: count, dtype: int64
The Flight Distance is a numerical column.
count    129880.000000
mean       1981.409055
std        1027.115606
min          50.000000
25%        1359.000000
50%        1925.000000
75%        2544.000000
max        6951.000000
Name: Flight Distance, dtype: float64
The Seat comfort is a numerical column.
count    129880.000000
mean          2.838597
std           1.392983
min           0.000000
25%           2.000000
50%           3.000000
75%           4.000000
max           5.000000
Name: Seat comfort, dtype: float64
The Departure/Arrival time convenient is a numerical column.
count    129880.000000
mean          2.990645
std           1.527224
min           0.000000
25%           2.000000
50%           3.000000
75%           4.000000
max           5.000000
Name: Departure/Arrival time convenient, dtype: float64
The Food and drink is a numerical column.
count    129880.000000
mean          2.851994
std           1.443729
min           0.000000
25%           2.000000
50%           3.000000
75%           4.000000
max           5.000000
Name: Food and drink, dtype: float64
The Gate location is a numerical column.
count    129880.000000
mean          2.990422
std           1.305970
min           0.000000
25%           2.000000
50%           3.000000
75%           4.000000
max           5.000000
Name: Gate location, dtype: float64
The Inflight wifi service is a numerical column.
count    129880.000000
mean          3.249130
std           1.318818
min           0.000000
25%           2.000000
50%           3.000000
75%           4.000000
max           5.000000
Name: Inflight wifi service, dtype: float64
The Inflight entertainment is a numerical column.
count    129880.000000
mean          3.383477
std           1.346059
min           0.000000
25%           2.000000
50%           4.000000
75%           4.000000
max           5.000000
Name: Inflight entertainment, dtype: float64
The Online support is a numerical column.
count    129880.000000
mean          3.519703
std           1.306511
min           0.000000
25%           3.000000
50%           4.000000
75%           5.000000
max           5.000000
Name: Online support, dtype: float64
The Ease of Online booking is a numerical column.
count    129880.000000
mean          3.472105
std           1.305560
min           0.000000
25%           2.000000
50%           4.000000
75%           5.000000
max           5.000000
Name: Ease of Online booking, dtype: float64
The On-board service is a numerical column.
count    129880.000000
mean          3.465075
std           1.270836
min           0.000000
25%           3.000000
50%           4.000000
75%           4.000000
max           5.000000
Name: On-board service, dtype: float64
The Leg room service is a numerical column.
count    129880.000000
mean          3.485902
std           1.292226
min           0.000000
25%           2.000000
50%           4.000000
75%           5.000000
max           5.000000
Name: Leg room service, dtype: float64
The Baggage handling is a numerical column.
count    129880.000000
mean          3.695673
std           1.156483
min           1.000000
25%           3.000000
50%           4.000000
75%           5.000000
max           5.000000
Name: Baggage handling, dtype: float64
The Checkin service is a numerical column.
count    129880.000000
mean          3.340807
std           1.260582
min           0.000000
25%           3.000000
50%           3.000000
75%           4.000000
max           5.000000
Name: Checkin service, dtype: float64
The Cleanliness is a numerical column.
count    129880.000000
mean          3.705759
std           1.151774
min           0.000000
25%           3.000000
50%           4.000000
75%           5.000000
max           5.000000
Name: Cleanliness, dtype: float64
The Online boarding is a numerical column.
count    129880.000000
mean          3.352587
std           1.298715
min           0.000000
25%           2.000000
50%           4.000000
75%           4.000000
max           5.000000
Name: Online boarding, dtype: float64
The Departure Delay in Minutes is a numerical column.
count    129880.000000
mean         14.713713
std          38.071126
min           0.000000
25%           0.000000
50%           0.000000
75%          12.000000
max        1592.000000
Name: Departure Delay in Minutes, dtype: float64
The Arrival Delay in Minutes is a numerical column.
count    129487.000000
mean         15.091129
std          38.465650
min           0.000000
25%           0.000000
50%           0.000000
75%          13.000000
max        1584.000000
Name: Arrival Delay in Minutes, dtype: float64
--------------------------------------------------
**Heat Map**
--------------------------------------------------

"""
