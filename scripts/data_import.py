from datasets import load_dataset
custom_path = "./data"
ds = load_dataset("moonworks/lunara-aesthetic", split="train[:1]", cache_dir=custom_path)
print(ds)

#读取数据类型
sample = ds[0]
print(type(sample))
print(ds.column_names)
print(sample)

import matplotlib.pyplot as plt
example = ds[0]

for field in ds.column_names:
    print(f"字段 {field} 的内容预览: {example[field]}")
