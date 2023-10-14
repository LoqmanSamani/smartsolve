
# Import specific classes/modules from each submodule
from .preprocessing import AnalyseData, SplitData, MissingValue, SelectFeature
from .models import LinearRegression, LogisticRegression, DecisionTree
from .evaluation import Validation


# Define package-level variables or configuration
PACKAGE_VERSION = "0.0.1"
DEFAULT_CONFIG = {
    "verbose": False,
    "debug": False,
}

# This line specifies what gets imported when using 'from learn-flow import *'
__all__ = [
    "AnalyseData",
    "SplitData",
    "MissingValue",
    "SelectFeature",
    "LinearRegression",
    "LogisticRegression",
    "Validation",
    "DecisionTree",
    "PACKAGE_VERSION",
    "DEFAULT_CONFIG",
]

