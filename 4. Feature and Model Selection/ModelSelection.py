import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


class ModelComparisonUtility:
    """
    Utility class to train, compare, and select the best classification model
    using GridSearchCV and ROC-AUC.
    """
    def quanQual(dataset):
        qual = []
        quan = []

        for columnName in dataset.columns:
            # TRUE numeric columns ONLY
            if pd.api.types.is_numeric_dtype(dataset[columnName]):
                quan.append(columnName)
            else:
                qual.append(columnName)
        return quan, qual
        
    def __init__(self, cv_splits=5, scoring="roc_auc", random_state=42):
        self.cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=random_state
        )
        self.scoring = scoring
        self.results_ = []
        self.best_model_ = None
        self.best_score_ = -1
        self.best_model_name_ = None
        self.feature_names_ = None   # ✅ NEW

    def _get_models(self):
        return {
            "Logistic Regression": {
                "estimator": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000))
                ]),
                "params": {
                    "model__C": [0.01, 0.1, 1, 10]
                }
            },

            "Random Forest": {
                "estimator": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 5, 10]
                }
            },

            "Gradient Boosting": {
                "estimator": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1]
                }
            },

            "SVM": {
                "estimator": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVC(probability=True))
                ]),
                "params": {
                    "model__C": [0.1, 1, 10],
                    "model__kernel": ["linear", "rbf"]
                }
            }
        }

    def fit(self, X, y):
        # ✅ store feature names ONCE
        self.feature_names_ = X.columns.tolist()

        models = self._get_models()

        for name, cfg in models.items():
            print(f"\nTraining {name}...")

            grid = GridSearchCV(
                estimator=cfg["estimator"],
                param_grid=cfg["params"],
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1
            )

            grid.fit(X, y)

            score = grid.best_score_

            self.results_.append({
                "Model": name,
                "ROC-AUC": score,
                "Best Params": grid.best_params_
            })

            if score > self.best_score_:
                self.best_score_ = score
                self.best_model_ = grid.best_estimator_
                self.best_model_name_ = name

        return self

    def save_best_model(self, path="loan_best_model_bundle.sav"):
        if self.best_model_ is None or self.feature_names_ is None:
            raise ValueError("Model or feature names missing.")

        bundle = {
            "model": self.best_model_,
            "features": self.feature_names_,
            "model_name": self.best_model_name_,
            "roc_auc": self.best_score_
        }

        joblib.dump(bundle, path)
        print(f"✅ Model + features saved to {path}")