import pandas as pd

path = '/opt/ml/code/train_label_fix.csv'

df = pd.read_csv(path)


def label_three(label):
    if label < 6:
        mask = 0
    elif 6 <= label < 12:
        mask = 1
    else:
        mask = 2
    if (label//3)%2 == 0:
        gender = 0
    else:
        gender = 1
    if label%3 == 0:
        age = 0
    elif label%3 == 1:
        age = 1
    else:
        age = 2
        
    return mask, gender, age 

df['mask'] = df['label'].map(lambda x: label_three(x)[0])
df['gender'] = df['label'].map(lambda x: label_three(x)[1])
df['age'] = df['label'].map(lambda x: label_three(x)[2])

df.to_csv('train_three_label.csv', index=False)
