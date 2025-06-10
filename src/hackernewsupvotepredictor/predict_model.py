import torch 
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanSquaredError
 
 
# load data
fea = torch.load("data/features.pt")
tar = torch.load("data/target.pt")


full_dataset = torch.utils.data.TensorDataset(fea,tar)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

# Hyperparameters
learning_rate = 0.001
batch_size = 512
num_epochs = 10
num_features = 1

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

def train_model():
    # Training setup  
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Model initialization
    model = HNUP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(5):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

train_model()

# tests
loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
x_test, y_test = next(iter(loader))   
preds = model.forward(x_test)
mse = MeanSquaredError()
mse(preds.squeeze(), y_test)






