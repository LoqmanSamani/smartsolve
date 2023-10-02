class FeatureScaling:

    """
    In this class, there are 9 different normalization methods for numeric data

    1) Min-max scaling
    2) Standardization (Z-score normalization)
    3) Robust scaling
    4) Max-Absolute scaling
    5) Power transformation
    6) Unit Vector Scaling (L2 Normalization)
    7) Log transformation
    8) Box-Cox transformation
    9) Yeo-Johnson transformation
    """

    def __init__(self, data, lam=None):
        self.data = data
        self.lam = lam



    def min_max_normalization(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            min_val = np.min(temporary_col)
            max_val = np.max(temporary_col)

            normalized_col = [(val - min_val)/(max_val - min_val) for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def z_score_normalization(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:
            temporary_col = self.data[col].values
            mean_val = np.mean(temporary_col)
            std_val = np.std(temporary_col)

            normalized_col = [(val - mean_val) / std_val  for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def robust_scaling(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            median_val = np.median(temporary_col)
            q1 = np.percentile(temporary_col,25)
            q3 = np.percentile(temporary_col, 75)
            iqr_val = q3 - q1

            normalized_col = [(val - median_val) / iqr_val  for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def max_absolute_scaling(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            max_abs = abs(np.max(temporary_col))

            normalized_col =[val / max_abs for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def power_transform(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values
            normalized_col = [np.power(val, self.lam) for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def unit_vector_scaling(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        norms = np.linalg.norm(self.data.values, axis=1, keepdims=True)

        for col in self.data.columns:
            temporary_col = self.data[col].values
            cal_norms = norms[0:len(self.data[col])]
            normalized_col = [val / norm for val, norm in zip(temporary_col, cal_norms)]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def log_transformation(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            normalized_col = [np.log(val) for val in temporary_col]
            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def box_cox_transformation(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values

            normalized_col = [np.power(val, self.lam - 1) if self.lam != 0
                              else np.log(val) for val in temporary_col]

            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)




    def yeo_johnson_transformation(self):

        import numpy as np
        import pandas as pd

        normalized_data = {}

        for col in self.data.columns:

            temporary_col = self.data[col].values
            normalized_col = []

            for val in temporary_col:

                if val > 0 and self.lam != 0:
                    normalized_col.append(np.power(val+1, self.lam-1) / self.lam)
                elif val < 0 and self.lam != 0:
                    normalized_col.append(-(np.power(-val+1, -self.lam+1) / self.lam))

                elif val < 0 and self.lam == 0:
                    normalized_col.append(np.log(val+1))

                elif val < 0 and self.lam == 0:
                    normalized_col.append(-(np.log(-val+1)))

                else:
                    normalized_col.append(val)

            normalized_data[col] = normalized_col

        return pd.DataFrame(normalized_data)
