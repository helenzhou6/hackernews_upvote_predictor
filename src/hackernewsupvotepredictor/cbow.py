import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import datetime

# logging seconds since start
starttime = datetime.datetime.now()

def log_seconds(now_datetime):
    delta =  now_datetime - starttime
    seconds = delta.total_seconds()
    return seconds

# set the dimension of embeddings
embedding_dim = 64
batch_size = 512

tests = ['anarchism','garden','production']

# read Wikipedia data and tokenize
with open("data/text8", "r") as f:
  raw_data = f.read()

# length of raw_data = 100,000,000
raw_data= raw_data[:100000]

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

print(f"Created vocab index, seconds since start of script: {log_seconds(datetime.datetime.now())}")

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
# Create TensorDataset and DataLoader
dataset = torch.utils.data.TensorDataset(context_tensor, target_tensor)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Created datasets, converted to tensor and created dataloader: {log_seconds(datetime.datetime.now())}")

# Model Architecture
class WordEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim, padding_idx=0)
        self.linear_1 = nn.Linear(in_features=embedding_dim, out_features=len(vocab))

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(axis=1)
        x = self.linear_1(x)
        return x


def print_mostsim(w, embeddings, top, skip_print = False):
    if skip_print:
        return
    f = embeddings[w]
    flen = torch.sqrt(torch.dot(f, f))
    simi = [torch.dot(x, f)/flen/ torch.sqrt(torch.dot(x, x)) if i!=w else -1000 for i, x in enumerate(embeddings)]
    ind = np.argpartition(np.array(simi), -top)[-top:]
    print(f'base word: {revvocab[w]}')
    for i in ind:
        print(revvocab[i])

# Training Setup
model = WordEmbeddings()
optimizer = optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training Loop
for epoch in range(20):
    train_loss = 0
    model.train()
    for feature, label in train_dataloader:
        model = model.to(device)
        feature = feature.to(device)
        label = label.to(device)

        y_train_pred = model(feature)

        loss = loss_fn(y_train_pred, label)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_dataloader)
    print(f"----- Epoch: {epoch} -----")
    print(f"Training Loss : {train_loss}")
    curemb = model.state_dict()['embeddings.weight']
    for word in tests:
        # Add skip_print=True to skip this print
        print_mostsim(vocab[word], curemb, 5) 

    print(f"Epoch finished, seconds since start of script: {log_seconds(datetime.datetime.now())}")
    print("\n")

# Save the embeddings
state = model.state_dict()
torch.save(state['embeddings.weight'], 'temp/wikipedia_embeddings.pt')
torch.save(state, 'temp/wikipedia_model_state.pth')
# model = WordEmbeddings()
# model.load_state_dict(torch.load('temp/wikipedia_model_state.pth'))

print(f"Model trained and weights etc saved, seconds since start of script: {log_seconds(datetime.datetime.now())}")

