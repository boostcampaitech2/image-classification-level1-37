import torch
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from train_nni_cutmix import seed_everything

class rand_bbox:
    def __init__(self,inputs, labels, beta1, beta2,random_seed):
        seed_everything(random_seed)
        self.labels = labels
        self.inputs = inputs
        self.size = inputs.size()
        self.beta1 = beta1
        self.beta2 = beta2
        self.get_lam()
        self.get_rand_index()
        self.get_targets()
        self.coordinate()

    def get_lam(self):
        self.lam = np.random.beta(self.beta1, self.beta2)
    
    def get_rand_index(self):
        self.rand_index = torch.randperm(self.size[0])

    def get_targets(self):
        self.target_a = self.labels
        self.target_b = self.labels[self.rand_index]

    def coordinate(self):
        W = self.size[2]
        H = self.size[3]
        cut_rat = np.sqrt(1. - self.lam)  # 패치 크기 비율
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # 패치의 중앙 좌표 값 cx, cy
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 패치 모서리 좌표 값
        self.x1 = 0
        # self.x1 = np.clip(cx - cut_w // 2, 0, H)
        self.y1 = np.clip(cy - cut_h // 2, 0, H)
        self.x2 = W
        # self.x2 = np.clip(cx + cut_w // 2, 0, H)
        self.y2 = np.clip(cy + cut_h // 2, 0, H)

    def get_cutmiximage_and_lam(self):
        self.inputs[:, :, self.x1:self.x2, self.y1:self.y2] = \
        self.inputs[self.rand_index, :, self.x1:self.x2, self.y1:self.y2]
        self.lam = 1 - ((self.x2 - self.x1) * (self.y2 - self.y1) / (self.inputs.size()[-1] * self.inputs.size()[-2]))
        return self.inputs, self.lam, self.target_a,self.target_b
    
class GetClassWeights:
    def __init__(self, labels):
        self.labels = labels

    def get_weights(self, weights_type):
        if weights_type == "weights_vanilla":
            class_weights = [v for l, v in sorted(Counter(self.labels).items())]
            return list(map(lambda x : 1-x/sum(class_weights),class_weights))
    
        elif weights_type == "class_weights_sklearn":
            le = LabelEncoder()
            y_ind = le.fit_transform(self.labels)
            if not all(np.in1d(np.unique(self.labels), le.classes_)):
                raise ValueError("classes should have valid labels that are in y")
            return len(self.labels) / (len(le.classes_) *np.bincount(y_ind).astype(np.float64))
        
        elif weights_type =="su":
            label = torch.tensor(self.labels)
            label = label.unsqueeze(0)
            label = label/torch.sum(label)
            weights = 1.0 / label
            return weights
        
        else:
            raise NameError