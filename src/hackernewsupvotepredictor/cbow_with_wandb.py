import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import wandb
import random
wandb.login()
 
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# set the dimension of embeddings
embedding_dim = 64 
# tests = ['anarchism','garden','production']

# read Wikipedia data and tokenize
with open("data/text8", "r") as f:
  raw_data = f.read()

# test = raw_data[:10000]

raw_data= raw_data[:1000000]

def tokenizer(text):
    # Stemming
    result = text.lower().replace("ing ", " ").replace("'s ", " ").replace("es ", "e ").replace("ied ", "y ")
    # Replace contents
    result = result.replace(", ", " <COMMA> ").replace(". ", " <FULLSTOP> ").replace("'", " <APOSTROPHE> ").replace("!", " <EXCLAMATION>").replace("? ", " <QUESTION> ").replace(": ", " <ELIPSES> ")
    result = result.replace(" aaaaaacceglllnorst ", " <UNKNOWN> ").replace(" aaaaaaccegllnorrst ", " <UNKNOWN> ").replace("  aaaaaah ", " <UNKNOWN> ").replace(" zzurf  ", " <UNKNOWN> ").replace(" zzum  ", " <UNKNOWN> ").replace(" ...  ", " ")
    result = result.replace(" and "," ").replace(" a "," ").replace(" the "," ")
    result = result.split(" ")
    result.remove("")
    return result

updtext = tokenizer(raw_data)

# create a vocabulary and encode words
revvocab = set(updtext)
revvocab.add("<UNKNOWN>")
revvocab = list(revvocab)
vocab = {word: index for index, word in enumerate(revvocab)}

# Save the vocabulary
with open("temp/vocabulary.json", "w") as fp:
    json.dump(vocab, fp)
# with open("temp/vocabulary.json", "r") as fp:
#     vocab = json.loads(fp)

# encode
encoded = [vocab.get(w, vocab["<UNKNOWN>"]) for w in updtext]

# rolling windows of 5 words
# Function to generate context generator for CBOW
def generate_cbow_data_context(sequence, context_size): 
    for j in range(context_size, len(sequence) - context_size):
        context = sequence[j - context_size: j] + sequence[j + 1: j + context_size + 1]
        yield context

# Function to generate target generator for CBOW
def generate_cbow_data_target(sequence, context_size):
    for j in range(context_size, len(sequence) - context_size):
        target = sequence[j]
        yield target

# Create datasets
context_data = list(generate_cbow_data_context(encoded, 2))
target_data = list(generate_cbow_data_target(encoded, 2))

# Convert to tensors
context_tensor = torch.tensor(context_data, dtype=torch.long)   # Shape: (N,2*context_size)
target_tensor = torch.tensor(target_data, dtype=torch.long)     # Shape: (N,)

# Split into train and test
randsel = np.random.random(size=len(context_tensor))

# Create TensorDataset and DataLoader
ds_test = torch.utils.data.TensorDataset(context_tensor[randsel>0.9], target_tensor[randsel>0.9])
ds_train = torch.utils.data.TensorDataset(context_tensor[randsel<=0.9], target_tensor[randsel<=0.9])

# Model Architecture
class WordEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)
        self.linear_1 = nn.Linear(in_features=embedding_dim, out_features=len(vocab))

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(axis=1)
        x = self.linear_1(x)
        return x

def train(model, train_loader, criterion, optimizer, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training Loop
    for epoch in range(config.epochs):
        train_loss = 0
        model.train()
        for feature, label in train_loader:
            model = model.to(device)
            feature = feature.to(device)
            label = label.to(device)

            y_train_pred = model(feature)

            loss = criterion(y_train_pred, label)
            train_loss = train_loss + loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        train_loss = train_loss / len(train_dataloader)
        wandb.log({"epoch": epoch, "loss": float(train_loss)})
        print(f"Epoch:{epoch} | Training Loss : {train_loss}") 
        
def test(model, test_loader):
    model.eval()
    
    correct, total = 0, 0
    for feature, label in test_loader:
        model = model.to(device)
        feature = feature.to(device)
        label = label.to(device)

        outputs = model(feature)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
     
        print(f"Accuracy of the model on the {total} " +
              f"contexts: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    # # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, feature, "model.onnx")
    # wandb.save("model.onnx")
    
    
#### wandb pieces
config = dict(
    epochs=20,
    batch_size=512,
    learning_rate=0.005,
    )

def make(config):
    # Make the data 
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=config.batch_size)

    # Make the model
    model = WordEmbeddings().to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer


def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="HNUP-CBOW", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model

model_pipeline(config)
 