"""Pipeline definitions for machine learning models."""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from inflamm_debate_fm.config.config import get_config


def get_linear_pipelines() -> dict[str, Pipeline]:
    """Get linear pipelines for Raw and Embedding data.

    Returns:
        Dictionary with 'Raw' and 'Embedding' keys containing Pipeline objects.
    """
    config = get_config()
    model_config = config.get("model", {})
    linear_config = model_config.get("linear", {})

    random_state = linear_config.get("random_state", 21)
    solver = linear_config.get("solver", "liblinear")
    max_iter = linear_config.get("max_iter", 5000)
    penalty = linear_config.get("penalty", "l2")

    raw = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver=solver, random_state=random_state, max_iter=max_iter, penalty=penalty
                ),
            ),
        ]
    )

    embedding = Pipeline(
        [
            (
                "clf",
                LogisticRegression(solver=solver, random_state=random_state, max_iter=max_iter),
            ),
        ]
    )

    return {"Raw": raw, "Embedding": embedding}


def get_nonlinear_pipelines() -> dict[str, Pipeline]:
    """Get nonlinear pipelines for Raw and Embedding data.

    Returns:
        Dictionary with 'Raw' and 'Embedding' keys containing Pipeline objects.
    """
    config = get_config()
    model_config = config.get("model", {})
    nonlinear_config = model_config.get("nonlinear", {})

    random_state = nonlinear_config.get("random_state", 21)
    kernel = nonlinear_config.get("kernel", "rbf")
    C = nonlinear_config.get("C", 1.0)
    gamma = nonlinear_config.get("gamma", "scale")
    probability = nonlinear_config.get("probability", True)

    raw = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel=kernel,
                    probability=probability,
                    C=C,
                    gamma=gamma,
                    random_state=random_state,
                ),
            ),
        ]
    )

    embedding = Pipeline(
        [
            (
                "clf",
                SVC(
                    kernel=kernel,
                    probability=probability,
                    C=C,
                    gamma=gamma,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {"Raw": raw, "Embedding": embedding}


def get_linear_pipeline_raw_only() -> Pipeline:
    """Get linear pipeline for Raw data only (for coefficient extraction).

    Returns:
        Pipeline object for Raw data.
    """
    config = get_config()
    model_config = config.get("model", {})
    linear_config = model_config.get("linear", {})

    random_state = linear_config.get("random_state", 21)
    max_iter = linear_config.get("max_iter", 5000)
    penalty = linear_config.get("penalty", "l2")
    solver = linear_config.get("solver", "lbfgs")

    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter, penalty=penalty, solver=solver, random_state=random_state
                ),
            ),
        ]
    )
