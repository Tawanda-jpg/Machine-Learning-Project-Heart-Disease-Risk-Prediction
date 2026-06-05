import os 
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    f1_score
)


def train_model():
    try:
        
        #load env file content to env variables
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

        DATASET_PATH = PROJECT_ROOT/ os.getenv("DATASET_DIR")/ os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT/ os.getenv("MODEL_DIR")/ os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT/ os.getenv("LOG_DIR")/ os.getenv("LOG_NAME")

        TARGET_COL = os.getenv("TARGET_COL")
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
        TEST_SIZE = float(os.getenv("TEST_SIZE"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        #LOADING DATA
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset with shape {df.shape} is loaded")
        
        #separate X and y
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        #create a signature for each feature row to prevent duplicate leakage to test data
        row_signature = pd.util.hash_pandas_object(X, index= False)

        #Group-based-split
        gss = GroupShuffleSplit(
            n_splits = 1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        train_idx, test_idx = next(gss.split(X, y, groups = row_signature))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logging.info(f"Train shape: {X_train.shape} and Test shape: {X_test.shape}")

        #best parameters from notebook tuning
        best_rfc = RandomForestClassifier(
             random_state=RANDOM_STATE,
             n_jobs = -1,
             bootstrap=True,
             ccp_alpha=0.0017,
             max_depth=5,
             max_features="sqrt",
             max_samples=0.6,
             min_samples_leaf=11,
             min_samples_split=30,
             n_estimators=1119
        )

        #keep scaler in pipeline to match notebook structure
        pipeline = Pipeline(
             steps=[
                  ('scaler', StandardScaler()),
                  ('model', best_rfc)
             ]
        )

        pipeline.fit(X_train, y_train)
        logging.info("Model Training Completed")

        #EVALUATION
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logging.info(f"Training Accuracy: {train_acc:.4f} | Training Recall:{train_recall:.4f} | Train F1 Score: {train_f1:.4f}")
        logging.info(f"Testing Accuracy:{test_acc:.4f} | Testing Recall:{test_recall:.4f} | Testing F1 Score:{test_f1:.4f}")

        logging.info("Train Classification Report:\n + classification_report(y_train, y_train_pred)")
        logging.info("Test Classification Report\n + classification_report(y_test, y_test_pred)")

        #SAVING TRAINED MODEL
        dump(pipeline, MODEL_PATH)
        logging.info(f"Model is saved to {MODEL_PATH}")

        logging.info("Training Script Completed")

    except Exception as e:
        print(f"Training Failed: {e}")
        logging.exception("Training Error Failed: {e}")
        raise

if __name__== "__main__":
        train_model()
