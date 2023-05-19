import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import mlflow

from lib.utils import (
    parse_config,
    log_results,
    load_dict,
    save_dict,
    save_model,
    METRICS,
    MODELS,
)


mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('vasenkov_hw')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def train_model(model, x, y):
    model.fit(x, y)

def train():
    config = parse_config('train')
    model_type = config['model']

    task_dir = 'data/train'
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    data = load_dict('data/prepare/data.json')
    train_x = data['train_x']
    test_x = data['test_x']
    train_y = data['train_y']
    test_y = data['test_y']
    x, y = train_x + test_x, train_y + test_y

    model = MODELS[model_type]()
    train_model(model, train_x, train_y)
    preds = model.predict(x)

    save_data = {
        'model_type': model_type,
    }
    save_dict(save_data, os.path.join(task_dir, 'data.json'))

    metrics = {}
    for metric_name in parse_config('eval')['metrics']:
        metrics[metric_name] = METRICS[metric_name](y, preds)
    save_dict(metrics, os.path.join(task_dir, 'metrics.json'))

    sns.heatmap(pd.DataFrame(train_x).corr())
    plt.savefig('data/train/heatmap.png')

    save_model(model_type, model)

    params = {}
    for i in parse_config().values():
        params.update(i)

    params['run_type'] = 'train'

    print(f'train params - {params}')
    print(f'train metrics - {metrics}')

    log_results(
        params=params,
        metrics=metrics,
        report=(y, preds),
        artifact='data/train/heatmap.png',
        model=(model_type, model)
    )

if __name__ == '__main__':
    train()
