import pandas as pd

dataset = pd.read_parquet('/home/test1267/test-6/qzk/Datasets/deepscaler/deepscaler.parquet')
print(dataset.iloc()[0])
dataset2 = pd.read_parquet('/home/test/test06/qzk/Datasets/deepscaler/deepscaler.parquet')
print(dataset2.iloc()[0])