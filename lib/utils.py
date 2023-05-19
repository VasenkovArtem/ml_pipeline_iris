import json
import pickle
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from functools import partial
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

import mlflow
from mlflow.sklearn import log_model as log_sklearn
from mlflow.xgboost import log_model as log_xgboost
from mlflow.catboost import log_model as log_catboost


METRICS = {
    'recall': partial(recall_score, average='macro'),
    'precision': partial(precision_score, average='macro'),
    'accuracy': accuracy_score,
}

MODELS = {
    'LogisticRegression': LogisticRegression,
    'SVC': SVC,
    'KNeighborsClassifier': KNeighborsClassifier,
    'GaussianProcessClassifier': GaussianProcessClassifier,
    'MLPClassifier': MLPClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'ExtraTreesClassifier': ExtraTreesClassifier,
    'AdaBoostClassifier': AdaBoostClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
    'XGBClassifier': XGBClassifier,
    'CatBoostClassifier': CatBoostClassifier
}

LOG_MODEL = {
    'LogisticRegression': log_sklearn,
    'SVC': log_sklearn,
    'KNeighborsClassifier': log_sklearn,
    'GaussianProcessClassifier': log_sklearn,
    'MLPClassifier': log_sklearn,
    'DecisionTreeClassifier': log_sklearn,
    'RandomForestClassifier': log_sklearn,
    'ExtraTreesClassifier': log_sklearn,
    'AdaBoostClassifier': log_sklearn,
    'GradientBoostingClassifier': log_sklearn,
    'HistGradientBoostingClassifier': log_sklearn,
    'XGBClassifier': log_xgboost,
    'CatBoostClassifier': log_catboost
}


def parse_config(stage: str = None) -> dict:
    with open('params.yaml', 'r') as f:
        params_data = yaml.safe_load(f)
    return params_data[stage] if stage else params_data

def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)

def save_model(model_type: str, model):
    if model_type == 'XGBClassifier':
        model.save_model('data/train/model.json')
    elif model_type == 'CatBoostClassifier':
        model.save_model('data/train/model.cbm')
    else:
        with open('data/train/model.pkl', 'wb') as f:
            pickle.dump(model, f)

def load_model(model_type: str):
    if model_type == 'XGBClassifier':
        model = XGBClassifier()
        model.load_model('data/train/model.json')
        return model
    if model_type == 'CatBoostClassifier':
        return CatBoostClassifier().load_model('data/train/model.cbm')
    with open('data/train/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def log_results(params=None, metrics=None, report=None, artifact=None,
                model=None):
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if report:
        target_names = ['setosa', 'versicolor', 'virginica']
        mlflow.log_text(
            classification_report(*report, target_names=target_names),
            'info/classification_report.txt'
        )
        mlflow.log_dict(
            classification_report(*report, target_names=target_names,
                                  output_dict=True),
            'info/classification_report.json'
        )
    if artifact:
        mlflow.log_artifact(artifact, artifact_path='info')
    if model:
        LOG_MODEL[model[0]](model[1], artifact_path='model')
