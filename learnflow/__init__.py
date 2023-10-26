from .preprocessing import AnalyseData, SplitData, MissingValue, SelectFeature, CategoricalData, FeatureScaling

from .models import (LinearRegression, LogisticRegression, DecisionTree, RandomForest, KMeansClustering,
                     KNearestNeighbors, NaiveBayes, PrincipalComponentAnalysis, SupportVectorMachines,
                     GradientBoosting, GaussianMixtureModel, SingularValueDecomposition)

from .evaluation import Validation


PACKAGE_VERSION = "1.0.0"
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

