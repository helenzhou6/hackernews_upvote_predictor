import torch
import torch.nn as nn
import torch.nn.functional as F

class CombineFeats(nn.Module):
    def __init__(self, feat_dim):
        super(CombineFeats, self).__init__()
        # Combine before feeding into model
        # combined_inputs = torch.stack([user_days_ft], dim=1)
        # num_features = combined_inputs.shape[1]  # like feat1, feat2, feat3
        # Simulated float features for each of the 3 sources
        # x_input = torch.randn(batch_size, num_features, feat_dim)
        # print(combined_inputs.shape)  # torch.Size([10, 1])

        feat_len, hidden_dim, do_rate = 1, 3, 0.2
        # expected data shape
        # Input tensor shape: [batch_size, num_features, feat_dim]
        # self.lin1 = nn.Linear(feat_dim, hidden_dim) # projects each feature
        self.lin1 = nn.Linear(1, hidden_dim) # projects each feature
        # self.lin2 = nn.Linear(hidden_dim * feat_len, 1) # combine features
        self.lin2 = nn.Linear(hidden_dim, 1) # combine features
        self.dropout = nn.Dropout(do_rate)

    def forward(self, x):
        # x = torch.permute(x, [1,0])
        # x = torch.unsqueeze(x)
        # x shape: [batch_size, num_feat, feat_dim]
        x = self.lin1(x) # shape: [batch_size, num_feat, hidden_dim], 500, 1, 
        x = F.relu(x) 
        # x = x.view(x.size(0), -1) # flatten: [batch_size, num_feat * hidden_dim]
        x = self.dropout(x)
        x = self.lin2(x) # final output layer
        return x

def combine_feat(user_days_data):
    user_days_ft = torch.permute(user_days_data, (1, 0))
    return user_days_ft