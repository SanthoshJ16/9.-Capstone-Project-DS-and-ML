import pandas as pd
import numpy as np

from sklearn.model_selection import (cross_val_score,GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (SelectKBest, chi2, RFE, SequentialFeatureSelector)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (MinMaxScaler,StandardScaler)
from sklearn.pipeline import Pipeline

from collections import defaultdict


class BestFeature:

    # ------------------------------
    # Load Dataset
    # ------------------------------
    @staticmethod
    def importDataset():
        return pd.read_csv("Loan_Prediction_Cleaned.csv")

    # ------------------------------
    # Evaluate Features
    # ------------------------------
    @staticmethod
    def evaluate_features(X, y, features, cv=5):
        model = BestFeature.logistic_pipeline()
    
        scores = cross_val_score(
            model,
            X[features],
            y,
            cv=cv,
            scoring="accuracy"
        )
        return scores.mean()

    # ------------------------------
    # SelectKBest
    # ------------------------------
    @staticmethod
    def select_k_best(X, y, k=5):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        skb = SelectKBest(score_func=chi2, k=k)
        skb.fit(X_scaled, y)

        features = X.columns[skb.get_support()].tolist()
        score = BestFeature.evaluate_features(X, y, features)

        return features, score

    # ------------------------------
    # RFE
    # ------------------------------
    @staticmethod
    def rfe_selection(X, y, k=5):
        model = BestFeature.logistic_pipeline()
    
        rfe = RFE(
            estimator=model,
            n_features_to_select=k,
            importance_getter=lambda est: est.named_steps["clf"].coef_
        )
    
        rfe.fit(X, y)
    
        features = X.columns[rfe.support_].tolist()
        score = BestFeature.evaluate_features(X, y, features)
    
        return features, score

    # ------------------------------
    # Feature Importance (Tree-Based)
    # ------------------------------
    @staticmethod
    def feature_importance_selection(X, y, k=5):
        rf = RandomForestClassifier(
            n_estimators=200, random_state=42
        )
        rf.fit(X, y)

        importances = pd.Series(
            rf.feature_importances_, index=X.columns
        )

        features = importances.nlargest(k).index.tolist()
        score = BestFeature.evaluate_features(X, y, features)

        return features, score

    # ------------------------------
    # Forward Selection
    # ------------------------------
    @staticmethod
    def forward_selection(X, y, k=5):
        model = BestFeature.logistic_pipeline()
    
        sfs = SequentialFeatureSelector(
            model,
            n_features_to_select=k,
            direction="forward",
            scoring="accuracy",
            cv=5
        )
    
        sfs.fit(X, y)
    
        features = X.columns[sfs.get_support()].tolist()
        score = BestFeature.evaluate_features(X, y, features)
    
        return features, score
    # ------------------------------
    # Backward Selection
    # ------------------------------
    @staticmethod
    def backward_selection(X, y, k=5):
        model = BestFeature.logistic_pipeline()
    
        sfs = SequentialFeatureSelector(
            model,
            n_features_to_select=k,
            direction="backward",
            scoring="accuracy",
            cv=5
        )
    
        sfs.fit(X, y)
    
        features = X.columns[sfs.get_support()].tolist()
        score = BestFeature.evaluate_features(X, y, features)
    
        return features, score

    # ------------------------------
    # Compare All Methods
    # ------------------------------
    @staticmethod
    def find_best_feature_selection(X, y, k=5):

        methods = {
            "SelectKBest": BestFeature.select_k_best,
            "RFE": BestFeature.rfe_selection,
            "Feature Importance": BestFeature.feature_importance_selection,
            "Forward Selection": BestFeature.forward_selection,
            "Backward Selection": BestFeature.backward_selection,
        }

        results = {}

        for name, method in methods.items():
            features, score = method(X, y, k)
            results[name] = {
                "features": features,
                "cv_accuracy": score
            }

        return (
            pd.DataFrame(results)
            .T
            .sort_values("cv_accuracy", ascending=False)
        )


    #### Preprocess Data for one hot encoding (Split Categorical and Numerical data)
    @staticmethod
    def preprocess_data(df, target_column, drop_columns=None):
        drop_columns = drop_columns or []
        
        X = df.drop(columns=[target_column] + drop_columns)
        y = df[target_column]
        
        # Identify categorical & numeric columns
        categorical_cols = X.select_dtypes(include="object").columns
        numeric_cols = X.select_dtypes(exclude="object").columns
        
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(
            X,
            columns=categorical_cols,
            drop_first=True
        )
        
        return X_encoded, y

    ### Create Logisting Pipeline to handle iterations issue and Scalar
    @staticmethod
    def logistic_pipeline():
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    n_jobs=-1
                ))
            ]
        )
    #### Feature Stability utility code
    def build_feature_stability_table(results_df):
        """
        Build a feature stability table from feature-selection results.
    
        Parameters:
            results_df (pd.DataFrame):
                Output of find_best_feature_selection()
    
        Returns:
            pd.DataFrame: feature stability table
        """
    
        feature_counter = defaultdict(int)
        feature_methods = defaultdict(list)
    
        for method, row in results_df.iterrows():
            for feature in row["features"]:
                feature_counter[feature] += 1
                feature_methods[feature].append(method)
    
        stability_df = pd.DataFrame({
            "Feature": feature_counter.keys(),
            "Frequency": feature_counter.values(),
            "Methods_Selected": [
                ", ".join(feature_methods[f])
                for f in feature_counter.keys()
            ]
        })
    
        stability_df = stability_df.sort_values(
            by=["Frequency", "Feature"],
            ascending=[False, True]
        ).reset_index(drop=True)
    
        return stability_df
    # ------------------------------
    # Finding Best K value
    # ------------------------------
    def tune_k_select_k_best(X, y, k_range, cv=5):
    
        results = []
    
        # chi2 requires non-negative values
        scaler_chi2 = MinMaxScaler()
        X_scaled = scaler_chi2.fit_transform(X)
    
        for k in k_range:
            selector = SelectKBest(score_func=chi2, k=k)
            X_selected = selector.fit_transform(X_scaled, y)
    
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000))
            ])
    
            scores = cross_val_score(
                model, X_selected, y,
                cv=cv, scoring="accuracy"
            )
    
            results.append({
                "k": k,
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std()
            })
    
        return pd.DataFrame(results)