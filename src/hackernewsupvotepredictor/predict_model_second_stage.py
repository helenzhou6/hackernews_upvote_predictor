import torch 
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError
from torch.utils.data import DataLoader

# load data
fea = torch.load("data/features.pt")
tar = torch.load("data/target.pt")
   
# Hyperparameters
learning_rate = 0.02
batch_size = 512
num_epochs = 5
num_features = fea.shape[1] 

# model
class HNUP(nn.Module):
    def __init__(self,  num_features = num_features): 
       super().__init__()
       self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
    def forward(self, x):
        p = self.linear_relu_stack(x)
        return p
        
model = HNUP()

# preparation
# loader
full_dataset = torch.utils.data.TensorDataset(fea,tar)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1]) 
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
criterion = nn.MSELoss() # this is for torch.float data
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
print("train session")
model.train()
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (features_b, target_b) in enumerate(train_dataloader):

        optimizer.zero_grad()
        outputs = model(features_b)

        loss = criterion(outputs.squeeze(), target_b)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    mean_loss = torch.stack(losses).mean()
    print(f'Epoch {epoch+1}, Loss: {mean_loss.item():.4f}') 

print("test session")
with torch.no_grad():
    model.eval()
    losses = []
    for batch_idx, (features_b, target_b) in enumerate(test_dataloader): 
 
        outputs = model(features_b)
        loss = criterion(outputs.squeeze(), target_b)
        losses.append(loss)
    mean_loss = torch.stack(losses).mean()
    print(f'Loss: {mean_loss.item():.4f}')

       