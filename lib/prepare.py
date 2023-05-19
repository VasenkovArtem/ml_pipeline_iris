import os

from sklearn import datasets
from sklearn.model_selection import train_test_split

from lib.utils import parse_config, save_dict, log_results

import mlflow


# а нужно ли логировать prepare stage в mlflow?
mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('vasenkov_hw')

def prepare_data():

    config = parse_config('prepare')
    features = config['features']
    test_size = config['test_size']

    task_dir = 'data/prepare'
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    iris = datasets.load_iris(as_frame=True)

    # артефактом таски "Предобработка фичей" является
    # "*Готовый* для обучения датасет"
    # а потому сделаем выбор нужных фичей здесь

    x = iris['data'][features].values.tolist()
    y = iris['target'].tolist()

    train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                        test_size=test_size)

    save_data = {
        'train_x': train_x,
        'test_x': test_x,
        'train_y': train_y,
        'test_y': test_y,
    }

    save_dict(save_data, os.path.join(task_dir, 'data.json'))

    params = {'run_type': 'prepare'}
    # сохраним только параметры prepare stage
    # (есть ли смысл сохранять model, metrics и тп, если мы готовим данные?)
    params.update(config)

    print(f'prepare params - {params}')

    # сохраним датасет - надо ли это делать в mlflow?
    log_results(
        params=params,
        artifact='data/prepare/data.json'
    )

if __name__ == '__main__':
    prepare_data()
