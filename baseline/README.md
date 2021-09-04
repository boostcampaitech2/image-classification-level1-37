# pstage_01_image_classification
## 사진을 보고 사진에 있는 사람의 나이, 성별, 마스크착용 유무를 판단하는 classification입니다. 이를 18개의 클래스로 나누어 출력합니다.

## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`
- train파일은 일반적인 train을 하는 train_val.py과 pseudo labeling을 하는 train_pseudo.py, 그리고 kfold로 데이터를 나눠 train하는 train_kfold.py 파일이 있습니다.
- train_val.py파일의 경우는 --cutmix를 true로 하면 cutmix를 사용할 수 있습니다.
- 모델을 불러올 때는 모델의 이름과 모델이 있는 라이브러리, 모델의 버전을 입력하면 해당 모델을 가져올 수 있습니다.
### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`
- inference.py에는 kfold로 나눠 학습한 모델를 앙상블해서 결과를 낼 수 있는 kfold_inference 함수가 있으니 kfold train을 한 경우 함수를 바꿔서 사용해주세요 
### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`
-
### 최종모델
- Model description(T2159)
https://towering-silkworm-c2e.notion.site/Model-description-240a220ea93d44f3b35f7d86ec74caab
- Pretrained model(tf_efficientnet_b3,5,6)
https://drive.google.com/drive/folders/19s_9OqbmkHvm7WLBiK1pry8SecQ1JIGr?usp=sharing

- Model description(T2243)
https://big-cut-d0b.notion.site/Model-description-4534bffb61ce4728be7d45f0afb82eef

