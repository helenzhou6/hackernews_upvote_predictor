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
num_words = 25000

# read Wikipedia data and tokenize
with open("data/text8", "r") as f:
  raw_data = f.read()

# tokenize
def tokenizer(text):
    # Stemming
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
 
# replace rare words with unknowns
def replace_rare(wordlist, limit):
    uni, counts = np.unique(wordlist, return_counts=True)
    unks  = set(uni[counts<limit])
    return ["<UNKNOWN>" if w in unks else w for w in wordlist]

"""
fulltext = replace_rare(tokenizer(raw_data), 5)
with open("temp/wikipedia_tokenized.json", 'w') as f:
    json.dump(fulltext, f, indent=2)
"""

with open("temp/wikipedia_tokenized.json", 'r') as f:
    fulltext = json.load(f)

# pick a sample
updtext = fulltext[:num_words]

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
#        train_loss = 0
        model.train()
        for feature, label in train_loader:
            model = model.to(device)
            feature = feature.to(device)
            label = label.to(device)

            y_train_pred = model(feature)

            loss = criterion(y_train_pred, label)
 #           train_loss = train_loss + loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

 #       print(f"Epoch:{epoch} | Training Loss : {train_loss}") 
    print("Training is done")
      
def test(model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    acc = correct / total
    return acc


def make(config):
    # Make the data 
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=config.batch_size)

    # Make the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WordEmbeddings().to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer


def objective(config):

   # make the model, data, and optimization problem
    model, train_loader, test_loader, criterion, optimizer = make(config)

    # and use them to train the model
    train(model, train_loader, criterion, optimizer, config)

    # and test its final performance
    return test(model, test_loader)

def main():
    run = wandb.init(config=wandb.config)
    accuracy = objective(wandb.config)
    wandb.log({"accuracy": accuracy})
    run.finish()

sweep_configuration = { 
    "name": "new_sweep",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"values": [16, 32, 64, 128, 256]},
        "epochs": {"values": [10, 15, 20]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="HNUP-CBOW-SWEEP", entity="ayzor-mlx")

wandb.agent(sweep_id, function=main, count=15)

