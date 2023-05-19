import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mlflow

from lib.utils import (
    parse_config,    
    load_dict,
    save_dict,
    load_model,
    log_results,
    METRICS
)

# если оставить только в train.py, eval stage не логируется (
mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('vasenkov_hw')

def eval():
    config = parse_config('eval')

    data_prepare = load_dict('data/prepare/data.json')

    data_train = load_dict('data/train/data.json')
    model_type = data_train['model_type']

    model = load_model(model_type)

    preds = model.predict(data_prepare['test_x'])

    if not os.path.exists('data/eval'):
        os.mkdir('data/eval')

    metrics = {}
    for metric_name in config['metrics']:
        metrics[metric_name] = METRICS[metric_name](data_prepare['test_y'],
                                                    preds)

    save_dict(metrics, 'data/metrics.json')

    sns.heatmap(pd.DataFrame(data_prepare['test_x']).corr())
    plt.savefig('data/eval/heatmap.png')

    params = {'run_type': 'eval'}
    for i in parse_config().values():
        params.update(i)

    print(f'eval params - {params}')
    print(f'eval metrics - {metrics}')

    log_results(
        params=params,
        metrics=metrics,
        report=(data_prepare['test_y'], preds),
        artifact='data/eval/heatmap.png',
        model=(model_type, model)
    )

if __name__ == '__main__':
    eval()
