from ._version import __version__
from .data import CorrelationFilter
from .model_selection import (
    check_assumptions,
    check_best_model,
    find_n_best_models,
    get_best_hparams,
    get_best_model,
    get_cv_performance,
    get_optimised_cv_performance,
    perform_statistical_tests,
    run_CV,
    tukey_hsd,
)
from .models import (
    CLASSIFIER_PARAMS,
    CLASSIFIERS,
    NON_PROBABILISTIC_MODELS,
    REGRESSOR_PARAMS,
    REGRESSORS,
)

__all__ = [
    "__version__",
    "CorrelationFilter",
    "check_assumptions",
    "tukey_hsd",
    "find_n_best_models",
    "perform_statistical_tests",
    "check_best_model",
    "get_cv_performance",
    "run_CV",
    "get_optimised_cv_performance",
    "get_best_hparams",
    "get_best_model",
    "CLASSIFIERS",
    "CLASSIFIER_PARAMS",
    "NON_PROBABILISTIC_MODELS",
    "REGRESSORS",
    "REGRESSOR_PARAMS",
]
