import torch
import numpy as np
from collections import Counter
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder

class rand_bbox:
    def __init__(self, size,lam):
        self.size = size
        self.lam = lam

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
        bbx1 = 0
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = W
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


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
        
        else:
            raise NameError