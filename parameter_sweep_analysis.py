import os
import re
import pandas as pd

files = [f for f in os.listdir("./output/experiments/") if "sweep" in f]
files = [f for f in files if "unlabel_percent" not in f]

columns = ["embedding_head", "training_step", "gamma", "loss_balance", "lambda_2", "mAP@IoU=0.5", "training_time (s)"]

rows = []
for filename in files:
    row = [re.search(".*eh-(\d)-s_(\d+)-g_(.*)-lb_(.*)-l2_(.*).txt", filename).group(i) for i in range(1,6)]

    with open(os.path.join("./output/experiments/", filename), 'r') as f:
        
        train_time = None
        mAP = None

        for line in f:
            rt = re.search("real\t(.*)m(.*)s.*", line)
            rm = re.search(".*mAP@0.50 (.*) mAP@0.55.*", line)

            if rt and not train_time:
                train_time = float(rt.group(1)) * 60 + float(rt.group(2))
            elif rm and not mAP:
                mAP = rm.group(1)

        row.append(mAP)
        row.append(train_time)

    f.close()

    rows.append(row)

df = pd.DataFrame(rows, columns=columns)
df = df.dropna()
df.to_csv("parameter_sweep_summary.csv", index=False)
