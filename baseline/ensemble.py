import os
import pandas as pd
from collections import Counter
import pdb

"""다양한 모델로 추출한 결과를 앙상블 할 수 있는 파일입니다. inference.py에서 만든 csv파일들을 조합합니다."""
def ensemble(file_path, ensemble_csv_list):
    #read csv files
    data = pd.read_csv(os.path.join(file_path, "info.csv"))
    data_ensemble = {}
    for i in range(len(ensemble_csv_list)):
        data_ensemble[i] = pd.read_csv(os.path.join(file_path, ensemble_csv_list[i]))

    #har voting
    all_predictions = []
    for i in range(len(data)):
        outputs = []
        for key in data_ensemble.keys():
            outputs.append(data_ensemble[key].ans[i])
        ans = Counter(outputs).most_common(1)
        all_predictions.append(ans[0][0])

    data["ans"] = all_predictions

    #save submission csv
    data.to_csv("/".join([file_path, "info.csv"]), index=False)
    print("Done!!")


if __name__=="__main__":
    file_path = "./output"
    ensemble_csv_list = ["submission_1.csv",
                         "submission_2.csv",
                         "submission_3.csv",
                         "submission_4.csv",
                         "submission_5.csv",
                         "submission_6.csv",
                         ]
    ensemble(file_path, ensemble_csv_list)