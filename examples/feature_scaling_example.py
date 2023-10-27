import numpy as np
import pandas as pd
from smartsolve.preprocessing import FeatureScaling


# Create a fake dataset with random data
np.random.seed(0)  # Set a random seed for reproducibility
data = pd.DataFrame({
    'Numeric1': np.random.rand(100),  # Random values between 0 and 1
    'Numeric2': np.random.randint(1, 100, size=100),  # Random integers between 1 and 100
    'Numeric3': np.random.normal(0, 1, 100),  # Random values from a normal distribution
    'Category1': np.random.choice(['A', 'B', 'C'], size=100),  # Categorical data
    'Category2': np.random.choice(['X', 'Y', 'Z'], size=100)  # Categorical data
})


# Print the original dataset
print('Original Data:')
print(data)


# Create an instance of the FeatureScaling class
example = FeatureScaling()


# Perform Min-Max normalization on specific columns
min_max = example.min_max(data=data, columns=['Numeric1', 'Numeric2', 'Numeric3'])
print('Min-Max Normalization:')
print(min_max)


# Perform Z-Score normalization on all numeric columns
z_score = example.z_score(data=data)
print('Z-Score Normalization:')
print(z_score)


# Perform Robust normalization on a specific column (Numeric3)
robust = example.robust(data=data, columns=['Numeric3'])
print('Robust Normalization:')
print(robust)


# Perform Absolute Max normalization on specific columns (Numeric2 and Numeric3)
abs_max = example.abs_max(data=data, columns=['Numeric2', 'Numeric3'])
print('Absolute Max Normalization:')
print(abs_max)


# Perform Power Transformation with a specified lambda value on specific columns
pow_transform = example.pow_transform(data=data, lam=4, columns=['Numeric1', 'Numeric2', 'Numeric3'])
print('Power Transformation:')
print(pow_transform)


# Perform Unit Vector Scaling (L2 Normalization) on specific columns (Numeric2 and Numeric3)
unit_vector = example.unit_vector(data=data, columns=['Numeric2', 'Numeric3'])
print('Unit Vector Scaling (L2 Normalization):')
print(unit_vector)


# Perform Log Transformation on a specific column (Numeric2)
log_transform = example.log_transform(data=data, columns=['Numeric2'])
print('Log Transformation:')
print(log_transform)


# Perform Box-Cox normalization with a specified lambda value (lam=3)
box_cox = example.box_cox(data=data, lam=3)
print('Box-Cox Normalization:')
print(box_cox)


# Perform Yeo-Johnson normalization on specific columns (Numeric1, Numeric2, and Numeric3) with a specified lambda value (lam=2)
yeo_johnson = example.yeo_johnson(data=data, lam=2, columns=['Numeric1', 'Numeric2', 'Numeric3'])
print('Yeo-Johnson Normalization:')
print(yeo_johnson)


"""
Outputs:
----------------------------------------------------
Original Data:
----------------------------------------------------
    Numeric1  Numeric2  Numeric3 Category1 Category2
0   0.548814         3 -0.769916         B         Z
1   0.715189         4  0.539249         A         X
2   0.602763        95 -0.674333         A         X
3   0.544883        99  0.031831         C         Z
4   0.423655        14 -0.635846         B         Y
..       ...       ...       ...       ...       ...
95  0.183191         4  0.063262         C         Z
96  0.586513        32  0.156507         C         Z
97  0.020108        10  0.232181         A         Z
98  0.828940        11 -0.597316         B         Y
99  0.004695        28 -0.237922         A         Y

[100 rows x 5 columns]
----------------------------------------------------
Min-Max Normalization:
----------------------------------------------------
    Numeric1  Numeric2  Numeric3 Category1 Category2
0   0.553146  0.010309  0.315526         B         Z
1   0.722283  0.020619  0.599723         A         X
2   0.607991  0.958763  0.336276         A         X
3   0.549151  1.000000  0.489571         C         Z
4   0.425911  0.123711  0.344631         B         Y
..       ...       ...       ...       ...       ...
95  0.181458  0.020619  0.496395         C         Z
96  0.591471  0.309278  0.516636         C         Z
97  0.015668  0.082474  0.533064         A         Z
98  0.837921  0.092784  0.352995         B         Y
99  0.000000  0.268041  0.431013         A         Y

[100 rows x 5 columns]
----------------------------------------------------
Z-Score Normalization:
----------------------------------------------------
    Numeric1  Numeric2  Numeric3 Category1 Category2
0   0.263681 -1.557596 -0.774031         B         Z
1   0.840771 -1.522843  0.636085         A         X
2   0.450811  1.639611 -0.671077         A         X
3   0.250048  1.778620  0.089539         C         Z
4  -0.170443 -1.175321 -0.629623         B         Y
..       ...       ...       ...       ...       ...
95 -1.004512 -1.522843  0.123394         C         Z
96  0.394445 -0.549780  0.223828         C         Z
97 -1.570183 -1.314330  0.305338         A         Z
98  1.235325 -1.279578 -0.588122         B         Y
99 -1.623641 -0.688789 -0.201014         A         Y

[100 rows x 5 columns]
----------------------------------------------------
Robust Normalization:
----------------------------------------------------
    Numeric1  Numeric2  Numeric3 Category1 Category2
0   0.548814         3 -0.523146         B         Z
1   0.715189         4  0.435181         A         X
2   0.602763        95 -0.453177         A         X
3   0.544883        99  0.063744         C         Z
4   0.423655        14 -0.425005         B         Y
..       ...       ...       ...       ...       ...
95  0.183191         4  0.086752         C         Z
96  0.586513        32  0.155008         C         Z
97  0.020108        10  0.210403         A         Z
98  0.828940        11 -0.396800         B         Y
99  0.004695        28 -0.133719         A         Y

[100 rows x 5 columns]
----------------------------------------------------
Absolute Max Normalization:
----------------------------------------------------
    Numeric1  Numeric2  Numeric3 Category1 Category2
0   0.548814  0.030303 -0.323067         B         Z
1   0.715189  0.040404  0.226276         A         X
2   0.602763  0.959596 -0.282959         A         X
3   0.544883  1.000000  0.013357         C         Z
4   0.423655  0.141414 -0.266810         B         Y
..       ...       ...       ...       ...       ...
95  0.183191  0.040404  0.026546         C         Z
96  0.586513  0.323232  0.065672         C         Z
97  0.020108  0.101010  0.097426         A         Z
98  0.828940  0.111111 -0.250642         B         Y
99  0.004695  0.282828 -0.099835         A         Y

[100 rows x 5 columns]
--------------------------------------------------------
Power Transformation:
--------------------------------------------------------
        Numeric1  Numeric2  Numeric3 Category1 Category2
0   9.071919e-02        81  0.351377         B         Z
1   2.616280e-01       256  0.084559         A         X
2   1.320041e-01  81450625  0.206774         A         X
3   8.814823e-02  96059601  0.000001         C         Z
4   3.221429e-02     38416  0.163459         B         Y
..           ...       ...       ...       ...       ...
95  1.126212e-03       256  0.000016         C         Z
96  1.183342e-01   1048576  0.000600         C         Z
97  1.634693e-07     10000  0.002906         A         Z
98  4.721635e-01     14641  0.127297         B         Y
99  4.860921e-10    614656  0.003204         A         Y

[100 rows x 5 columns]
--------------------------------------------------------
Unit Vector Scaling (L2 Normalization):
--------------------------------------------------------
    Numeric2  Numeric3  Numeric1 Category1 Category2
0   0.968611 -0.248583  0.548814         B         Z
1   0.991035  0.133604  0.715189         A         X
2   0.999975 -0.007098  0.602763         A         X
3   1.000000  0.000322  0.544883         C         Z
4   0.998970 -0.045371  0.423655         B         Y
..       ...       ...       ...       ...       ...
95  0.999875  0.015814  0.183191         C         Z
96  0.999988  0.004891  0.586513         C         Z
97  0.999731  0.023212  0.020108         A         Z
98  0.998529 -0.054222  0.828940         B         Y
99  0.999964 -0.008497  0.004695         A         Y

[100 rows x 5 columns]
----------------------------------------------------
Log Transformation:
----------------------------------------------------
    Numeric1  Numeric2  Numeric3 Category1 Category2
0   0.548814  1.098612 -0.769916         B         Z
1   0.715189  1.386294  0.539249         A         X
2   0.602763  4.553877 -0.674333         A         X
3   0.544883  4.595120  0.031831         C         Z
4   0.423655  2.639057 -0.635846         B         Y
..       ...       ...       ...       ...       ...
95  0.183191  1.386294  0.063262         C         Z
96  0.586513  3.465736  0.156507         C         Z
97  0.020108  2.302585  0.232181         A         Z
98  0.828940  2.397895 -0.597316         B         Y
99  0.004695  3.332205 -0.237922         A         Y

[100 rows x 5 columns]
----------------------------------------------------
Box-Cox Normalization:
----------------------------------------------------
    Numeric1  Numeric2  Numeric3
0   0.301196         9  0.592771
1   0.511496        16  0.290790
2   0.363324      9025  0.454725
3   0.296898      9801  0.001013
4   0.179483       196  0.404300
..       ...       ...       ...
95  0.033559        16  0.004002
96  0.343997      1024  0.024494
97  0.000404       100  0.053908
98  0.687142       121  0.356786
99  0.000022       784  0.056607

[100 rows x 3 columns]
--------------------------------
Yeo-Johnson Normalization:
--------------------------------
    Numeric1  Numeric2  Numeric3
0   0.774407       2.0 -0.282499
1   0.857595       2.5  0.769625
2   0.801382      48.0 -0.298626
3   0.772442      50.0  0.515915
4   0.711827       7.5 -0.305652
..       ...       ...       ...
95  0.591596       2.5  0.531631
96  0.793256      16.5  0.578253
97  0.510054       5.5  0.616091
98  0.914470       6.0 -0.313025
99  0.502348      14.5 -0.403903

[100 rows x 3 columns]

"""
