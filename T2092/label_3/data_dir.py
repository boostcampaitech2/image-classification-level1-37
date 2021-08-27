import os
import pandas as pd
class cfg_train:
    data_dir = '/opt/ml/input/data/train'
    img_dir = '/opt/ml/input/data/train/images'
    df_path = 'train.csv'
class cfg_test:
    data_dir = '/opt/ml/input/data/eval'
    img_dir = '/opt/ml/input/data/eval/images'
    df_path = 'info.csv'

if __name__ == "__main__":
    submission = pd.read_csv(os.path.join(cfg_test.data_dir, 'info.csv'))[:5]
    label = [[2,5,2,4,3],[1,1,1,1,1],[1,1,1,1,1]]
    class_sum = list(map(lambda x,y,z: 6*x+3*y+z, label[0],label[1],label[2]))
    submission['ans'] = class_sum
    print(submission)
