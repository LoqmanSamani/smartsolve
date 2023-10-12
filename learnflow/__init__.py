
# Import specific classes/modules from each sub-module
from .preprocessing import SplitData
from .preprocessing import MissingValue
#from .models import LinearRegression
#from .evaluation import Validation



# Define package-level variables or configuration
PACKAGE_VERSION = "0.0.1"
DEFAULT_CONFIG = {
    "verbose": False,
    "debug": False,
}

# This line specifies what gets imported when using 'from my_package import *'
__all__ = [
    "SplitData",
    "MissingValue",
    #"LinearRegression",
    #"Validation",
    "PACKAGE_VERSION",
    "DEFAULT_CONFIG",
]

