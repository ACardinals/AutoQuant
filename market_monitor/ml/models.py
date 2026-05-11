from __future__ import annotations


def available_models() -> list[str]:
    return ["hist_gradient_boosting", "logistic_regression", "random_forest"]


def create_model(name: str, random_state: int = 0):
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=random_state)
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state))
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    available = ", ".join(available_models())
    raise ValueError(f"Unknown ML model '{name}'. Available: {available}")
