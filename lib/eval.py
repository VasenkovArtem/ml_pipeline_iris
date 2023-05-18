import os.path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import mlflow

from lib.utils import (
    parse_config,    
    load_dict,
    save_dict,
    load_model,
    log_results,
    METRICS,
    LOG_MODEL
)

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('vasenkov_hw')

def eval():
    config = parse_config('eval')

    data = load_dict('data/prepare/data.json')
    model_type = data['model_type']

    model = load_model(model_type)

    preds = model.predict(data['test_x'])

    if not os.path.exists('data/eval'):
        os.mkdir('data/eval')

    metrics = {}
    for metric_name in config['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['test_y'], preds)

    save_dict(metrics, 'data/metrics.json')

    sns.heatmap(pd.DataFrame(data['test_x']).corr())
    plt.savefig('data/eval/heatmap.png')

    params = {'run_type': 'eval'}
    for i in params_data.values():
        params.update(i)

    print(f'eval params - {params}')
    print(f'eval metrics - {metrics}')

    log_results(
        params=params,
        metrics=metrics,
        report=(data['test_y'], preds),
        artifact='data/eval/heatmap.png',
        model=(model_type, model))

if __name__ == '__main__':
    eval()
