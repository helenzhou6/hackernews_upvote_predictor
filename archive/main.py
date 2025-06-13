# from hackernewsupvotepredictor.download_cbow_rawdata import download_data_from_url
from hackernewsupvotepredictor.cbow import WordEmbeddings, train_cbow_model, Tokenizer, tokenizer
from hackernewsupvotepredictor.combined_feats import CombineFeats, combine_feat
from hackernewsupvotepredictor.download_hn_title_data import download_hn_title_data
from hackernewsupvotepredictor.all_steps_combined import CombinedAllModel
# from hackernewsupvotepredictor.predict_model import HNUP
from hackernewsupvotepredictor.predict_model2 import get_final_pred
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from datasets import Dataset
import pandas as pd
import numpy as np

# def pad_collate(batch):
#   (xx, yy) = zip(*batch)
#   x_lens = [len(x) for x in xx]
#   y_lens = [len(y) for y in yy]

#   xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
#   yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

#   return xx_pad, yy_pad, x_lens, y_lens

# some of them had a seq_len around 120
def pad_collate(batch, max_len=150):
    # Extract each field into separate lists
    titles = [item["title"] for item in batch]
    # user_days = [item["user_days"] for item in batch]
    # labels = [item["label"] for item in batch]

    # # Get original lengths (if needed)
    # title_lens = [len(x) for x in titles]

    # Pad titles to max_len
    titles_padded = []
    for seq in titles:
        padded = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) if len(seq) < max_len else seq
        titles_padded.append(padded)

    titles_padded = torch.stack(titles_padded)  # (batch_size, 200)

    # # Pad sequences
    # titles_padded = pad_sequence(titles, batch_first=True, padding_value=0,)
    # titles_padded = [...titles_padded ]

    # # Stack scalar labels into a tensor
    # labels = torch.stack(labels)

    # Convert user_days and label to tensors
    user_days = torch.tensor([item["user_days"] for item in batch], dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1)

    return {
        "title": titles_padded,
        "user_days": user_days,
        "label": labels,
        # "title_lens": title_lens,
    }

def main():
    # set the dimension of embeddings
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

#     # Loading the embeddings into title CBOW model
#     # download_hn_title_data()  # to download the title data if not already there
#     with open("data/hn_title_data", "r") as f:
#         hn_title_data = f.read()
#     hn_title_data = hn_title_data[:1_000_000]
#     title_model = WordEmbeddings(wiki_tokenizer, embedding_dim)
#     # title_model.load_state_dict(torch.load('temp/wikipedia_embeddings.pt', weights_only=True))

#     ### Combine the features into a combined ML model
#     # Num of days user has been around as first feature - ML predict output. Output is FloatTensor
#     user_days_raw = torch.load('temp/user_days_output.pt')

#     all_data = pd.read_csv("data/all_data.csv")
#     title_data = all_data["title"]
#     title_tokenizer = Tokenizer(title_data.to_csv(), "temp/title_vocab.json")
#     title_cbow_model = WordEmbeddings(title_tokenizer, embedding_dim)
#     # print(title_cbow_model)
#     # training cbow for title
#     all_data["title_encoding"] = title_data.astype('str').apply(lambda title_data: [title_tokenizer.vocab[word] if title_tokenizer.vocab.get(word, None) else title_tokenizer.vocab.get("<UNKNOWN>") for title in title_data for word in tokenizer(title)])

#     ds = Dataset.from_dict({
#         "title": all_data["title_encoding"], 
#         'user_days': all_data["diff"], 
#         "label": all_data["score"]
#         }
#         ).with_format("torch")

#     train_ds, test_ds = torch.utils.data.random_split(ds, [0.9, 0.1])
#     train_dataloader = DataLoader(train_ds, batch_size=4, collate_fn=pad_collate)
#     test_dataloader = DataLoader(test_ds, batch_size=4, collate_fn=pad_collate)

#     # define models
#     pred_ft_model = get_final_pred
#     combined_ft_model = CombineFeats(500)

#     for param in title_cbow_model.parameters():
#         param.requires_grad = False

#     base_models = [title_cbow_model]
#     models = CombinedAllModel(
#         base_models, 
#         combined_ft_model, 
#         # pred_ft_model
#         )

#     # lr_rate = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
#     # save_final_score = {}
#     # for lr in lr_rate:
#     #     print(f"lr is {lr}")
#     # save_final_score[lr] = []
#     lr=0.01
#     criterion = nn.MSELoss() # this is for torch.float data
#     optimizer = optim.Adam(models.parameters(), lr=lr)

#     # train
#     print("train session")
#     for epoch in range(num_epochs):
#         losses = []
#         for batch_idx, data in enumerate(train_dataloader):
#             titles_b = data["title"] # [4,100] or [batch, word_lengh]
#             users_b = data["user_days"] # [4,1] or [batch, single_score]
#             target_b = data["label"] # [4,1] or [batch, single_score]

#             data_b = (titles_b, users_b)

#             optimizer.zero_grad()
#             outputs = models(data_b)
#             pred = get_final_pred(outputs)  # should depend on model output

#             # loss = criterion(outputs, target_b)
#             loss = criterion(pred, target_b)
            
#             loss.backward()
#             optimizer.step()
#             losses.append(loss)
#         mean_loss = torch.stack(losses).mean()
#         print(f'Epoch {epoch+1}, Loss: {mean_loss.item():.4f}')
#         # save_final_score[lr].append(np.round(loss.item()))
    
#     # print(save_final_score)

#         # test_input_format_cbow_model(cbow_model, title_tokenizer, batch_size)    
#         # user_days_raw = hacker_news_data["user_account_date"]
#         # print(f"combimed_data shape: {combimed_data.shape}")
#         # combined_out = combined_ft_model.forward(combimed_data)
#         # print(f"combined_out shape: {combined_out.shape}")
#         # pred_score = pred_ft_model.forward(combined_out)
#         # print(f"Prediction score shape: {pred_score.shape}")

#         # models_tmp = CombinedAllModel()

#         # # Step 1: Initialize model
#         # models = models_tmp(
#         #     base_models=[model1, model2, model3],
#         #     combined_model=model4,
#         #     predict_model=model5
#         # )

#     print("test session")
#     with torch.no_grad():
#         models.eval()
#         losses = []
#         for batch_idx, data in enumerate(test_dataloader):
#             titles_b = data["title"] # [4,100] or [batch, word_lengh]
#             users_b = data["user_days"] # [4,1] or [batch, single_score]
#             target_b = data["label"] # [4,1] or [batch, single_score]
#             # target_b = target_b.unsqueeze(1)  # shape: [4, 1]

#             data_b = (titles_b, users_b)
#             outputs = models(data_b)
#             loss = criterion(outputs, target_b)
#             losses.append(loss)
#         mean_loss = torch.stack(losses).mean()
#         print(f'Loss: {mean_loss.item():.4f}')

# # # optimizer
# # # loss_func

# # ### during training

# # # Step 2: Freeze base models
# # for param in combined_model.base_models.parameters():
# #     param.requires_grad = False

# # # Step 3: Define optimizer (important: after freezing)
# # optimizer = torch.optim.Adam(
# #     filter(lambda p: p.requires_grad, combined_model.parameters()),
# #     lr=1e-3
# # )


if __name__ == "__main__":
    main()
