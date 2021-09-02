import os
import pandas as pd
from collections import Counter

def ensemble(file_path):
    #read csv files
    data = pd.read_csv("/".join([file_path, "info.csv"]))
    data1 = pd.read_csv("/".join([file_path, "submission_1.csv"])) # efficientnet_b5(tf)
    data2 = pd.read_csv("/".join([file_path, "submission_2.csv"])) # efficientnet_b3
    data3 = pd.read_csv("/".join([file_path, "submission_3.csv"]))
    data4 = pd.read_csv("/".join([file_path, "submission_4.csv"]))
    data5 = pd.read_csv("/".join([file_path, "submission_5.csv"]))
    data6 = pd.read_csv("/".join([file_path, "submission_6.csv"]))

    #har voting
    all_predictions = []
    for i in range(len(data)):
        outputs = [
                   data1.ans[i], 
                   data2.ans[i],
                   data3.ans[i],
                   data4.ans[i],
                   data5.ans[i],
                   data6.ans[i],
                  ]
        ans = Counter(outputs).most_common(1)
        all_predictions.append(ans[0][0])

    data["ans"] = all_predictions

    #save submission csv
    data.to_csv("/".join([file_path, "info.csv"]), index=False)


if __name__=="__main__":
    file_path = "./output"
    ensemble(file_path)