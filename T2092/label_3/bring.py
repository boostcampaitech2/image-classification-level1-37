


from data_dir import cfg_test
import os
import torch
import pandas as pd
from eval import eval


mask_model_path = os.path.join(cfg_test.data_dir,'checkpoints','model_3', 'max_model_mask_7_0.025.pt')
gender_model_path = os.path.join(cfg_test.data_dir,'checkpoints','model_3','max_model_gender_4_0.027.pt')
age_model_path = os.path.join(cfg_test.data_dir,'checkpoints','model_3','max_model_age_7_0.051.pt')

mask_model = torch.load(mask_model_path)['model']
gender_model = torch.load(gender_model_path)['model']
age_model = torch.load(age_model_path)['model']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} is using")
for myModel in [mask_model,gender_model,age_model]:
    myModel.to(device)
ans = eval([mask_model,gender_model,age_model],device)
submission = pd.read_csv(os.path.join(cfg_test.data_dir, 'info.csv'))
submission['ans'] = ans
submission.to_csv(os.path.join(cfg_test.data_dir, 'answer3.csv'), index=False)
print('test is done!')