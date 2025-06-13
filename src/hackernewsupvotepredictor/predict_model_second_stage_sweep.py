import torch 
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError
from torch.utils.data import DataLoader
import wandb
import random
wandb.login()
 
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# load data
fea = torch.load("data/features.pt")
tar = torch.load("data/target.pt")
   
# Hyperparameters
num_features = fea.shape[1] 

# model
class HNUP(nn.Module):
    def __init__(self, num_hidden, num_features = num_features): 
       super().__init__()
       self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1),
        )
    def forward(self, x):
        p = self.linear_relu_stack(x)
        return p

# preparation
# loader
full_dataset = torch.utils.data.TensorDataset(fea,tar)
ds_train, ds_test = torch.utils.data.random_split(full_dataset, [0.9, 0.1])

def train(model, train_loader, criterion, optimizer, config):
    model.train()
    for epoch in range(config.epochs):
        losses = []
        for batch_idx, (features_b, target_b) in enumerate(train_loader):

            optimizer.zero_grad()
            outputs = model(features_b)

            loss = criterion(outputs.squeeze(), target_b)
            loss.backward()
            optimizer.step()
            losses.append(loss)
        mean_loss = torch.stack(losses).mean()
    print('Training is done')
      
def test(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        losses = []
        for batch_idx, (features_b, target_b) in enumerate(test_loader): 
 
            outputs = model(features_b)
            loss = criterion(outputs.squeeze(), target_b)
            losses.append(loss)
        mean_loss = torch.stack(losses).mean()
        return mean_loss


def make(config):
    # Make the data 
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=config.batch_size)

    # Make the model 
    model = HNUP(num_hidden=config.num_hidden)

    # Make the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer


def objective(config):

   # make the model, data, and optimization problem
    model, train_loader, test_loader, criterion, optimizer = make(config)

    # and use them to train the model
    train(model, train_loader, criterion, optimizer, config)

    # and test its final performance
    return test(model, test_loader, criterion)

def main():
    run = wandb.init(config=wandb.config)
    ave_loss = objective(wandb.config)
    wandb.log({"ave_loss": ave_loss})
    run.finish()

sweep_configuration = { 
    "name": "new_sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "ave_loss"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"values": [16, 32, 64, 128, 256, 512]},
        "epochs": {"values": [10, 15, 20]},
        "num_hidden": {"values": [4,8,16,32]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="HNUP-2ND-SWEEP", entity="ayzor-mlx")

wandb.agent(sweep_id, function=main, count=15)
