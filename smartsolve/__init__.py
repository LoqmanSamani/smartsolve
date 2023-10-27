from smartsolve.preprocessing import AnalyseData
from smartsolve.preprocessing import SplitData
from smartsolve.preprocessing import MissingValue
from smartsolve.preprocessing import SelectFeature
from smartsolve.preprocessing import CategoricalData
from smartsolve.preprocessing import FeatureScaling
from smartsolve.models import LinearRegression
from smartsolve.models import LogisticRegression
from smartsolve.models import DecisionTree
from smartsolve.models import DecisionTreeNode
from smartsolve.models import RandomForest
from smartsolve.models import KMeansClustering
from smartsolve.models import KNearestNeighbors
from smartsolve.models import NaiveBayes
from smartsolve.models import PrincipalComponentAnalysis
from smartsolve.models import SupportVectorMachines
from smartsolve.models import GradientBoosting
from smartsolve.models import GaussianMixtureModel
from smartsolve.models import SingularValueDecomposition
from smartsolve.evaluation import Validation


PACKAGE_VERSION = "0.0.4"
DEFAULT_CONFIG = {
    "verbose": False,
    "debug": False,
}

__all__ = [
    "AnalyseData",
    "SplitData",
    "MissingValue",
    "SelectFeature",
    "CategoricalData",
    "FeatureScaling",
    "LinearRegression",
    "LogisticRegression",
    "DecisionTree",
    "DecisionTreeNode",
    "RandomForest",
    "KMeansClustering",
    "KNearestNeighbors",
    "NaiveBayes",
    "PrincipalComponentAnalysis",
    "SupportVectorMachines",
    "GradientBoosting",
    "GaussianMixtureModel",
    "SingularValueDecomposition",
    "Validation",
    "PACKAGE_VERSION",
    "DEFAULT_CONFIG",
]

