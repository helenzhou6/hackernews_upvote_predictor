
############################################################

### import packages

import torch
import torch.nn as nn
import torch.optim as optim

############################################################

# Example corpus
text = "we are learning natural language processing with pytorch".split()

############################################################

# Create vocabulary
vocab = list(set(text))
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}
vocab_size = len(vocab)

############################################################

### preprocess data into tokenizer

# Create context-target pairs (CBOW)
def make_context_target(text, window_size=2):
    data = []
    for i in range(window_size, len(text) - window_size):
        context = [text[i - j - 1] for j in range(window_size)] + [text[i + j + 1] for j in range(window_size)]
        target = text[i]
        data.append((context, target))
    return data

data = make_context_target(text, window_size=2)

############################################################

### cbow function

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        embeds = self.embeddings(context_idxs)  # shape: (context_len, embedding_dim)
        combined = embeds.mean(dim=0)  # average the embeddings
        out = self.linear1(combined)
        return out

############################################################

### training

EMBEDDING_DIM = 10
model = CBOW(vocab_size, EMBEDDING_DIM)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(21):
    total_loss = 0
    for context, target in data:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        target_idx = torch.tensor([word_to_ix[target]], dtype=torch.long)

        # Forward pass
        output = model(context_idxs)

        # Loss & backprop
        loss = loss_function(output.view(1, -1), target_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

############################################################
