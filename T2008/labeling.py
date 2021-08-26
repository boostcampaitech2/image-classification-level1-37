import os
import pandas as pd


class MakeLabel:
    """
        base train_csv 파일을 호출하여 train image 경로와 image label 데이터를 추가해줍니다.
        path : {train.csv 경로, train_label.csv 경로}
        만약 train_label.csv 파일이 존재하면 labeling()함수가 바로 종료됩니다.

    """
    def __init__(self, path, wrong_data=None):
        self.path = path
        self.wrong_data = wrong_data
        self.df = pd.read_csv(self.path["train_vanilla"])
        self.train_vanilla_path = os.path.split(self.path["train_vanilla"])[0]
        

    def fix_wrong_data(self):
        if "gender" in self.wrong_data.keys():
            wrong_gender = self.wrong_data["gender"]
            self.df["gender"] = self.df.gender.map(lambda x : self.change_gender(x) if x in wrong_gender else x)

            # self.df[wrong_gender_condition] = self.df[wrong_gender_condition].gender.map(self.change_gender)
        
        if "mask" in self.wrong_data.keys():
            wrong_mask, mask1, mask2 = self.wrong_data["mask"]
            self.df["mask_"] = self.df["mask_"].map(lambda x : x if not x in wrong_mask else mask1 if x == mask2 else mask2)

    def change_gender(self, x):
        return "male" if x=="female" else "female"


    def get_image_path(self):
        image_path_list = []
        folder_path = os.path.join(self.train_vanilla_path, "images")
        folder_list = os.listdir(folder_path)

        for folder in folder_list:
            if folder[:6].isdigit():
                image_path = os.path.join(folder_path, folder)
                image_list = os.listdir(image_path)
                image_list = list(filter(lambda x : not x.startswith("._") and not "checkpoints" in x, image_list))
                # mask = list(map(lambda x : os.path.splitext(x)[0],image_list))
                # image_list = list(map(lambda x : os.path.join(image_path,x),image_list))
                for image in image_list:
                    mask = os.path.splitext(image)[0]
                    image_full_path = os.path.join(image_path,image)
                    image_path_list.append((folder,mask,image_full_path))
        
        image_path_df = pd.DataFrame(image_path_list, columns=["path","mask_","img_path"])
        self.df = pd.merge(self.df,image_path_df, on="path")
        self.df.to_csv(os.path.join(self.train_vanilla_path,"train_with_labels.csv"))

    
    def get_features(self, info):
        mask,age,gender = info.mask_, info.age, info.gender
        # mask = mask.split("/")[-1]
        mask = "2" if "normal" in mask else "1" if "incorrect" in mask else "0"
        gender = "0" if gender=="male" else "1"
        age = "0" if age<30 else "1" if 30<=age<60 else "2"
        return mask+gender+age
    
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
        if os.path.exists(self.path["train_label"]):
            print("labeling된 csv 파일이 존재합니다.")
            print("파일명: train_label.csv","\n","파일경로: /opt/ml/input/data/train/train_with_labels_fix_wrong_data.csv")
        else:
            self.get_image_path()
            if self.wrong_data != None:
                self.fix_wrong_data()
            # self.df.to_csv(os.path.join(self.train_vanilla_path,"train_with_labels.csv"))
            self.df["label"] = self.df.apply(lambda x : self.get_image_label(x),axis=1)
            self.df.to_csv(os.path.join(self.train_vanilla_path,"train_with_labels_fix_wrong_data.csv"))
            print("labeling된 csv 파일 생성이 완료되었습니다.")
            print("파일명: train_with_labels_fix_wrong_data.csv","\n","파일경로: /opt/ml/input/data/train/train_with_labels_fix_wrong_data.csv")


# if __name__ == "__main__":
#     path = {"train_label" : '/opt/ml/input/data/train/train_with_labels.csv',
#                      "train_vanilla" : '/opt/ml/input/data/train/train.csv',
#                      "validation" : '/opt/ml/input/data/eval/info.csv'}
#     tmp = MakeLabel(path)
#     tmp.labeling()
    
    # """
    #     사용 예시
    #     train_csv_path = '/opt/ml/input/data/train'
    #     data_train = pd.read_csv(os.path.join(train_csv_path, "train.csv"))
    #     make_label_data = MakeLabel(data_train, train_csv_path)
    #     make_label_data.labeling()
    #     print("labeling End")
    # """


