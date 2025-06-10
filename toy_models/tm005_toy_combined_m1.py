
###################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

###################################################################

### set Model Architecture

class CombineFeats(nn.Module):
    def __init__(self, feat_len, feat_dim, hidden_dim, do_rate):
        super(CombineFeats, self).__init__()
        # expected data shape
        # Input tensor shape: [batch_size, num_features, feat_dim]
        self.lin1 = nn.Linear(feat_dim, hidden_dim) # projects each feature
        self.lin2 = nn.Linear(hidden_dim * feat_len, 1) # combine features
        self.dropout = nn.Dropout(do_rate)

    def forward(self, x):
        # x shape: [batch_size, num_feat, feat_dim]
        x = self.lin1(x) # shape: [batch_size, num_feat, hidden_dim]
        x = F.relu(x) 
        x = x.view(x.size(0), -1) # flatten: [batch_size, num_feat * hidden_dim]
        x = self.dropout(x)
        x = self.lin2(x) # final output layer
        return x

# initialize the NN
feat_len, feat_dim, hidden_dim, do_rate = 3, 100, 30, 0.2
model = CombineFeats(feat_len, feat_dim, hidden_dim, do_rate)
print(model)

###################################################################

### Case 1: Combine outside the model class

# This gives maximum flexibility, especially if:
# You may want to use different preprocessing logic.
# Each CBOW vector comes from a separate model or process.
# You might want to weight them, concatenate selectively, or visualize them before combining.

# combine cbow features data outside of the combined model

batch_size = 10
feat_dim = 100

# Assume each CBOW tensor shape: [batch_size, embed_dim]
cbow1 = torch.randn(10, 100)
cbow2 = torch.randn(10, 100)
cbow3 = torch.randn(10, 100)

# Combine before feeding into model
combined_inputs = torch.stack([cbow1, cbow2, cbow3], dim=1)  # Shape: [10, 3, 100]

num_features = combined_inputs.shape[1]  # like feat1, feat2, feat3

# Simulated float features for each of the 3 sources
# x_input = torch.randn(batch_size, num_features, feat_dim)
print(combined_inputs.shape)  # torch.Size([10, 1])

output = model(combined_inputs)
print(output.shape)  # torch.Size([10, 1])

# Good for:
# Custom input pipelines
# Mixing embeddings (e.g., from different models or layers)

###################################################################

### Case 2: Combine inside the model class

# Do this if:
# Your model owns the logic of how features are handled.
# You always pass in separate inputs, like: model(cbow1, cbow2, cbow3)
# You want cleaner high-level code.

class CBOWCombiner(nn.Module):
    def __init__(self, embed_dim, hidden_dim, final_dim, dropout_rate):
        super().__init__()
        self.lin = nn.Linear(embed_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim * 3, final_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, cbow1, cbow2, cbow3):
        h1 = F.relu(self.lin(cbow1))
        h2 = F.relu(self.lin(cbow2))
        h3 = F.relu(self.lin(cbow3))
        combined = torch.cat([h1, h2, h3], dim=1)  # shape: [batch_size, hidden_dim * 3]
        out = self.dropout(combined)
        return self.final(out)

model = CBOWCombiner(embed_dim=100, hidden_dim=50, final_dim=1, dropout_rate=0.2)
output = model(cbow1, cbow2, cbow3)

# Good for:
# Encapsulation
# Cleaner API
# When CBOW tensors are logically tied to model structure

###################################################################

