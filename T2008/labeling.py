import os
import pandas as pd
from tqdm.notebook import tqdm

class MakeLabel:
    def __init__(self, csv_file, base_path, drop_features):
        self.df = csv_file.drop(drop_features, axis=1)
        self.base_path = base_path

    def get_image_path(self):
        image_path_list = []
        folder_path = os.path.join(self.base_path, "images")
        fodler_list = os.listdir(folder_path)

        for folder in fodler_list:
            if folder[:6].isdigit():
                image_path = os.path.join(folder_path, folder)
                image_list = os.listdir(image_path)
                image_list = list(filter(lambda x : not x.startswith("._") and not "checkpoints" in x, image_list))
                image_list = list(map(lambda x : os.path.join(image_path,x),image_list))
                for path in image_list:
                    image_path_list.append((folder,path))
        
        image_path_df = pd.DataFrame(image_path_list, columns=["path","img_path"])
        self.df = pd.merge(self.df,image_path_df, on="path")
        self.df.to_csv(os.path.join(self.base_path,"train_with_labels.csv"))


    def get_features(self, info):
        mask_,age,gender = info.img_path, info.age, info.gender
        
        mask_ = mask_.split("/")[6]
        mask = "2" if "normal" in mask_ else "1" if "incorrect" in mask_ else "0"
        gender = "0" if gender=="male" else "1"
        age = "0" if age<30 else "1" if 30<=age<60 else "2"
        return mask + gender + age
    
    def get_image_label(self, info):
        """
            1st_number : mask(mask:0, incorrect:1, normal:2)
            2rd_number : gender(male:0, female:1)
            3rd_number : age(<30:0, 30<= <60:1, <60:2)
            
        """
        features = self.get_features(info)
        feature_list = ["000","001","002", "010","011","012","100","101","102","110","111","112", "200","201","202", "210","211","212"]
        return feature_list.index(features)

    
    def labeling(self):
        self.get_image_path()
        self.df["label"] = self.df.apply(lambda x : self.get_image_label(x),axis=1)
        self.df.to_csv(os.path.join(self.base_path,"train_with_labels.csv"))


if __name__ == "__main__":
    train_csv_path = "./input/data/train"
    data_train = pd.read_csv(os.path.join(train_csv_path, "train.csv"))
    drop_features = ["id","race"]
    make_label_data = MakeLabel(data_train, train_csv_path, drop_features)
    make_label_data.labeling()
    print("labeling End")


