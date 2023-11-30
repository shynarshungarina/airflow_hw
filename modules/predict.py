import pandas as pd
import os
import json
import dill
from datetime import datetime
from glob import glob

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    df_test = []
    data_files = [f'{path}/data/test/7310993818.json', f'{path}/data/test/7313922964.json',
                  f'{path}/data/test/7315173150.json', f'{path}/data/test/7316152972.json',
                  f'{path}/data/test/7316509996.json']
    name = [7310993818, 7313922964, 7315173150, 7316152972, 7316509996]
    for filename in data_files:
        with open(filename) as json_file:
            data = json.load(json_file)
            df = pd.DataFrame.from_dict([data])
            df_test.append(df)
    df_test = pd.concat(df_test)

    filename = glob(f'{path}/data/models/cars_pipe_*.pkl')[0]
    with open(filename, 'rb') as file:
        best_pipe = dill.load(file)

    pred = best_pipe.predict(df_test)

    pred_data = pd.DataFrame({'car_id': name, 'pred': pred},
                             columns=['car_id', 'pred'])
    pred_data.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv',
                     sep=',', index=False)


if __name__ == '__main__':
    predict()
