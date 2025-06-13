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

tests = ['anarchism','production']

# Tokenizing
def tokenizer(text):
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

def replace_rare(wordlist, limit):
    uni, counts = np.unique(wordlist, return_counts=True)
    unks  = set(uni[counts<limit])
    return ["<UNKNOWN>" if w in unks else w for w in wordlist]

class Tokenizer():
    def __init__(self, data, vocab_json_filename):
        self.data = data
        updtext = replace_rare(tokenizer(data), 5)

        # create a vocabulary and encode words
        revvocab = set(updtext)
        revvocab.add("<UNKNOWN>")
        self.revvocab = list(revvocab)
        self.vocab = {word: index for index, word in enumerate(revvocab)}

        # Save the vocabulary
        with open(vocab_json_filename, "w") as fp:
            json.dump(self.vocab, fp)

        print(f"Created vocab index, seconds since start of script: {log_seconds(datetime.datetime.now())}")

        # encode
        self.encoded = [self.vocab.get(w, self.vocab["<UNKNOWN>"]) for w in updtext]

    def print_mostsim(self, w, embeddings, top, skip_print = False):
        if skip_print:
            return
        f = embeddings[w]
        flen = torch.sqrt(torch.dot(f, f))
        simi = [torch.dot(x, f)/flen/ torch.sqrt(torch.dot(x, x)) if i!=w else -1000 for i, x in enumerate(embeddings)]
        ind = np.argpartition(np.array(simi), -top)[-top:]
        print(f'base word: {self.revvocab[w]}')
        for i in ind:
            print(self.revvocab[i])

# Model Architecture
class WordEmbeddings(nn.Module):
    def __init__(self, tokenizer, embedding_dim):
        super(WordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=len(tokenizer.vocab), embedding_dim=embedding_dim, padding_idx=0)
        self.linear_1 = nn.Linear(in_features=embedding_dim, out_features=len(tokenizer.vocab))
        self.tokenizer = tokenizer

    def forward(self, x): 
        x = self.embeddings(x) # input shape [batch, word_len] -> encoded input [batch, word_len, embed_dim]
        x = x.mean(axis=1) # encoded input [batch, word_len, embed_dim] -> mean enc [batch, embed_dim]
        x = self.linear_1(x) # mean enc [batch, embed_dim] -> lin transform [batch, num_emb]
        return x
    
# rolling windows of 5 words
 # Function to generate context generator for CBOW
def _generate_cbow_data_context(sequence, context_size): 
    for j in range(context_size, len(sequence) - context_size):
        context = sequence[j - context_size: j] + sequence[j + 1: j + context_size + 1]
        yield context

# Function to generate target generator for CBOW
def _generate_cbow_data_target(sequence, context_size):
    for j in range(context_size, len(sequence) - context_size):
        target = sequence[j]
        yield target

# Training Setup
# def test_input_format_cbow_model(model, tokenizer, batch_size):
#     # Create datasets
#     context_data = list(_generate_cbow_data_context(tokenizer.encoded, 2))
#     print(context_data)
#     print(f"context_data size {len(context_data)}, context_data dtype {type(context_data)}")

# Training Setup
def train_cbow_model(model, tokenizer, batch_size):
    # Create datasets
    context_data = list(_generate_cbow_data_context(tokenizer.encoded, 2))
    target_data = list(_generate_cbow_data_target(tokenizer.encoded, 2))

    # Convert to tensors
    context_tensor = torch.tensor(context_data, dtype=torch.long)   # Shape: (N,2*context_size)
    target_tensor = torch.tensor(target_data, dtype=torch.long)     # Shape: (N,)
    # Create TensorDataset and DataLoader
    dataset = torch.utils.data.TensorDataset(context_tensor, target_tensor)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Created datasets, converted to tensor and created dataloader: {log_seconds(datetime.datetime.now())}")


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

        with torch.no_grad():
            for word in tests:
                # Add skip_print=True to skip this print
                tokenizer.print_mostsim(tokenizer.vocab[word], curemb, 5, skip_print=True) 

        torch.cuda.empty_cache()
        print(f"Epoch finished, seconds since start of script: {log_seconds(datetime.datetime.now())}")
        print("\n")

    # Save the embeddings
    state = model.state_dict()
    torch.save(state['embeddings.weight'], 'temp/wikipedia_embeddings.pt')
    torch.save(state, 'temp/wikipedia_model_state.pth')
    # model = WordEmbeddings()
    # model.load_state_dict(torch.load('temp/wikipedia_model_state.pth'))

    print(f"Model trained and weights etc saved, seconds since start of script: {log_seconds(datetime.datetime.now())}")


# To actually run cbow.py - uncomment out the following code and run the file:
embedding_dim = 64
batch_size = 512
num_epochs = 5

# Run download_cbow_rawdata.py to download
with open("data/text8", "r") as f:
    wiki_data = f.read()
wiki_data= wiki_data[:100000]

# Training Setup
wiki_tokenizer = Tokenizer(wiki_data, "temp/vocabulary.json")
cbow_model = WordEmbeddings(wiki_tokenizer, embedding_dim)
# print(cbow_model)

train_cbow_model(cbow_model, wiki_tokenizer, batch_size)