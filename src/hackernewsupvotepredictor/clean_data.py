import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime

df = pd.read_parquet("data/submissions.parquet.gzip")
df.head()

# find differences in dates (inputs are datetimes)
def days_between(d1, d2): 
    return (d2 - d1).dt.days

# x = df.loc[0,"user_created"]
# y = df.loc[0,"time"]

# print(x)
# print(y)
# print(days_between(x,y))

def prep_features(data):
    dt = data.copy() 
    dt['diff'] = days_between(dt['user_created'], dt['time']) 
    a = dt[["diff"]].to_numpy()
    return dt, torch.Tensor(a)

d, fea = prep_features(df) 

def prep_target(data):
    r = torch.Tensor(data['score'].values)
    r = torch.log(torch.add(r, 1.0))
    return r
tar = prep_target(df) 

torch.save(fea, "data/features.pk")
torch.save(tar, "data/target.pk")

 