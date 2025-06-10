import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
# import matplotlib.pyplot as plt


'''
import urllib.request
url = 'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8'
response = urllib.request.urlopen(url)
data = response.read()      # a bytes object
text = data.decode('utf-8')
'''

# set the dimension of embeddings
embedding_dim = 16
batch_size = 1024

# read Wikipedia data and tokenize
with open("data/text8", "r") as f:
  raw_data = f.read()

# test = raw_data[:10000]

def tokenizer(text):
    # Stemming
    result = text.lower().replace("ing ", " ").replace("'s ", " ").replace("es ", "e ").replace("ied ", "y ")
    # Replace contents
    result = result.replace(", ", " <COMMA> ").replace(". ", " <FULLSTOP> ").replace("'", " <APOSTROPHE> ").replace("!", " <EXCLAMATION>").replace("? ", " <QUESTION> ").replace(": ", " <ELIPSES> ")
    result = result.replace(" aaaaaacceglllnorst ", " <UNKNOWN> ").replace(" aaaaaaccegllnorrst ", " <UNKNOWN> ").replace("  aaaaaah ", " <UNKNOWN> ").replace(" zzurf  ", " <UNKNOWN> ").replace(" zzum  ", " <UNKNOWN> ").replace(" ...  ", " ")
    return result.split(" ")

updtext = tokenizer(raw_data)
# unique, counts = np.unique(np.array(updtext), return_counts=True)

# plt.hist(counts)
# plt.show()

# encode words
vocab = {word: index for index, word in enumerate(set(updtext))}
defval = vocab["<UNKNOWN>"] 
encoded = [vocab.get(w, defval) for w in updtext]

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


# Training Setup
model = WordEmbeddings()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.95)
loss_fn = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training Loop
for epoch in range(11):
    train_loss = 0
    model.train()
    for feature, label in train_dataloader:
        model = model.to(device)
        feature = feature.to(device)
        label = label.to(device)

        y_train_pred = model(feature)

        loss = loss_fn(y_train_pred, label)
        train_loss = train_loss + loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_dataloader)
    print(f"Epoch:{epoch} | Training Loss : {train_loss}")

# Save the embeddings
state = model.state_dict()
torch.save(state['embeddings.weight'], 'temp/wikipedia_embeddings.pt')
torch.save(state, 'temp/wikipedia_model_state.pth')
# model = WordEmbeddings()
# model.load_state_dict(torch.load('temp/wikipedia_model_state.pth'))

# Save the vocabulary
with open("temp/vocabulary.txt", "w") as fp:
    json.dump(vocab, fp)
# with open("temp/vocabulary.txt", "r") as fp:
#     vocab = json.loads(fp)
