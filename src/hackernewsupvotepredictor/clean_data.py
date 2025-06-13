import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime
import json

# main dataset
df = pd.read_parquet("data/submissions.parquet.gzip")
df.head()

# elements necessary for using embeddings
with open("temp/vocabulary.json", "r") as fp:
    vocab = json.load(fp)

embs = torch.load('temp/wikipedia_embeddings.pt')            

# Tokenizing
def tokenizer(text):
    result = text.lower()
    conv ={",":"<COMMA>", "\.\.\.":" <ELLIPSES> ", "\.": " <FULLSTOP> ",
           "\'": " <APOSTROPHE> ", "!": " <EXCLAMATION> ", "\?": " <QUESTION> ", ":": " <COLON> ",}
    for ins, outs in conv.items():
        result = result.replace(ins, outs)
    result = result.split(" ") 
    result = [w for w in result if len(w)>0 and w not in ['and','the','a','of','in','to']] 
    stems = set(result)
    result = [ s[:-1] if s[-1]=="s" and s[:-1] in stems else s for s in result ]
    result = [ s[:-4] if s[-4:]=="ning" and s[:-4] in stems else s for s in result ]    
    result = [ s[:-4] if s[-4:]=="ming" and s[:-4] in stems else s for s in result ]   
    result = [ s[:-3] if s[-3:]=="ing" and s[:-3] in stems else s for s in result ]   
    result = [ s[:-3] + 'y' if s[-3:]=="ied" and s[:-3] + 'y' in stems else s for s in result ]
    result = [ s[:-2] if s[-2:]=="ed" and s[:-2] in stems else s for s in result ]   
    result = [ s[:-2]+'e' if s[-2:]=="ed" and s[:-2]+'e' in stems else s for s in result ]
    return result

def encode(wordlist):
    punct = ["<COMMA>", "<ELLIPSES>", "<FULLSTOP>", "<APOSTROPHE>","<EXCLAMATION>","<QUESTION>","<COLON>"]
    unkpunct  = [w for w in punct if w not in vocab]
    defval = vocab['<UNKNOWN>']
    return [vocab.get(w,defval) for w in wordlist if w not in unkpunct]

def title2emb(title):
    codes = encode(tokenizer(title))
    if len(codes)<1:
        return embs[vocab['<UNKNOWN>']]
    else:
        return embs[codes].mean(dim=0)

# find differences in dates (inputs are datetimes)
def days_between(d1, d2): 
    return (d2 - d1).dt.days

def prep_features(data):
    dt = data.copy()
    t = torch.empty((dt.shape[0], embs.shape[1]))
    for i, r in dt.iterrows():
        t[i] = title2emb(r['title'])
    dt['diff'] = days_between(dt['user_created'], dt['time'])
    return dt, torch.cat((t, torch.Tensor(dt[["diff"]].to_numpy())), dim=1)

d, fea = prep_features(df) 

def prep_target(data):
    r = torch.Tensor(data['score'].values)
    r = torch.log(torch.add(r, 1.0))
    return r

def prep_other_var(data, var_name):
    r = data[var_name]
    return r

tar = prep_target(df)

# # ['id','by','title','score','url','time','user_created','user_karma']
# def get_all_data(data):
#     dt = data.copy()
#     dt['diff'] = days_between(dt['user_created'], dt['time'])
#     df = dt[['title', 'diff', 'score']]
#     return df
# all_data = get_all_data(df)

torch.save(fea, "data/features.pt")
torch.save(tar, "data/target.pt")