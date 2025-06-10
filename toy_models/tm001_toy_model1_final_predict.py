
############################################################

### import packages

import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt

############################################################

### set variable parameters

# focus on score up to 500
thres = 500

###
###
###

############################################################

### load dataset

# load 4mil hacker_news dataset
# df = pd.read_parquet("hf://datasets/julien040/hacker-news-posts/story.parquet")
df = pd.read_parquet("../data/submissions.parquet.gzip")

df_non0 = df[df["score"]>0]["score"] # as there are negative scores
total_count = df_non0.shape[0]
max_count = df_non0.max()
min_count = df_non0.min()

print(f"sample size is {total_count}"); 
print(f"min_count is {min_count}"); 
print(f"max_count is {max_count}"); 

############################################################

### create a list of tuples (score, a proportion of score over 4 mil)

prop_data = []

for i in range(1,max_count+1):
    count_prob = df_non0[df_non0==i].shape[0]/total_count
    if count_prob > 0:
        # print(f"score-{i}: proportion is {count_prob}")
        prop_data.append((i, count_prob))

count_np = np.array([count for count, _ in prop_data])
prob_np = np.array([prob for _, prob in prop_data])

############################################################

### visualization of raw data

plt.plot(count_np, 1/prob_np); plt.savefig("../data/data.png"); plt.show(); 
# plt.plot(count_np[:thres], 1/prob_np[:thres]); plt.show(); 
# plt.plot(count_np[:thres], np.log(1/prob_np[:thres])); plt.show(); 
# plt.plot(count_np[thres:], 1/prob_np[thres:]); plt.show(); 
# plt.plot(count_np[thres:], np.log(1/prob_np[thres:])); plt.show(); 
# plt.plot(count_np[thres:], prob_np[thres:]); plt.show(); 

############################################################

### compute a toy formula to fit to data

# actual data with normalization
x_true = count_np[:thres]
y_true = np.log(1/prob_np[:thres])

# fitting curve

# Fit a polynomial of degree 1 (a linear function) to the data
t = np.log(x_true)
p = np.polyfit(t, y_true, 1)

# Construct the fitted curve
a = p[0] # it was around 1.6
b = p[1] # it was around 0.5
x_fitted = np.linspace(np.min(x_true), np.max(x_true), 100)
y_fitted = a * np.log(x_fitted) + b
# print(f"a {a}, b {b}")

# based on the above formula, predict x-val based on actual y-val
x_recovered = np.round(np.exp( (y_true - b)/a ),0)

# plt.plot(x_true, y_true); plt.plot(x_fitted, y_fitted); plt.show(); 

# compute loss
# mean squared error (mse)
mse = np.mean((x_true - x_recovered)**2)
print(f"mse is {mse}")

############################################################

# next step is 
# to use neural(linear) model to make more accurate performance

