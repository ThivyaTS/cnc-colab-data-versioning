
import pandas as pd

data_path = '/content/drive/MyDrive/cnc-project/data_v1_train/train.csv'
df = pd.read_csv(data_path)
print(df.describe())
