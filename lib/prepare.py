from sklearn import datasets
from lib.utils import parse_config, save_dict

def prepare_data():
    config = parse_config('prepare')
    features = config['features']
    test_size = config['test_size']

    task_dir = 'data/prepare'

    iris = datasets.load_iris(as_frame=True)

    # артефактом таски "Предобработка фичей" является "*Готовый* для обучения датасет"
    # а потому сделаем выбор нужных фичей здесь

    x = iris['data'][features].values.tolist()
    y = iris['target'].tolist()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)

    save_data = {
        'train_x': train_x,
        'test_x': test_x,
        'train_y': train_y,
        'test_y': test_y,
    }

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    save_dict(save_data, os.path.join(task_dir, 'data.json'))
