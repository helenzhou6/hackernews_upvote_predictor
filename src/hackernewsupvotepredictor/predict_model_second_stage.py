import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
wandb.login()

def load_data():
    fea = torch.load("data/features.pt")
    tar = torch.load("data/target.pt")
    return fea, tar

def define_model(num_features):
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

    model = HNUP(num_features)
    return model

def convert_to_dl(fea,tar, batch_size, train_rate=0.9):
    test_rate = 1-train_rate
    full_dataset = torch.utils.data.TensorDataset(fea,tar)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_rate,test_rate])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, test_dataloader

def train(model, num_epochs, train_dataloader, optimizer, criterion):
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
        if epoch % 50 == 0:
            wandb.log({"epoch": epoch, "loss": float(mean_loss.item())})
            print(f'Epoch {epoch+1}, Loss: {mean_loss.item():.4f}') 

def test(model, test_dataloader, criterion):
    print("test session")
    with torch.no_grad():
        model.eval()
        losses = []
        for batch_idx, (features_b, target_b) in enumerate(test_dataloader): 
    
            outputs = model(features_b)
            loss = criterion(outputs.squeeze(), target_b)
            losses.append(loss)
        mean_loss = torch.stack(losses).mean()
        wandb.log({"loss": float(mean_loss.item())})
        print(f'Loss: {mean_loss.item():.4f}')

def main(config):
    # tell wandb to get started
    with wandb.init(project="HNUP-CBOW", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        epochs=config["epochs"]
        batch_size=config["batch_size"]
        learning_rate=config["learning_rate"]

        fea, tar = load_data()
        num_features = fea.shape[1]
        train_dataloader, test_dataloader = convert_to_dl(fea,tar, batch_size, train_rate=0.9)
        model = define_model(num_features)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train(model, epochs, train_dataloader, optimizer, criterion)
        state = model.state_dict()
        torch.save(state, 'temp/final_model.pt')
        test(model, test_dataloader, criterion)

def reinstate_org_model():
        fea, tar = load_data()
        num_features = fea.shape[1]
        model = define_model(num_features)
        return model

config = dict(
    epochs=201,
    batch_size=512,
    learning_rate=0.02,    
    )

# main(config)



