from hackernewsupvotepredictor.download_cbow_rawdata import download_data_from_url
from hackernewsupvotepredictor.cbow import WordEmbeddings, train_cbow_model, Tokenizer
from hackernewsupvotepredictor.combined_feats import CombineFeats, combine_feat
import torch.nn as nn
import torch
import pandas as pd



def main():
    # set the dimension of embeddings
    embedding_dim = 64
    batch_size = 512

    # length of raw_data = 100,000,000
    # url = 'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8'
    # wiki_data = download_data_from_url(url)
    # truncated_wiki_data= wiki_data[:100000]

    with open("data/text8", "r") as f:
        wiki_data = f.read()

    wiki_data= wiki_data[:100000]

    tokenizer = Tokenizer(wiki_data)
    # # Training Setup
    cbow_model = WordEmbeddings(tokenizer, embedding_dim)
    train_cbow_model(cbow_model, tokenizer, batch_size)

    # Loading the embeddings into titile CBOW model
    title_model = WordEmbeddings(tokenizer, embedding_dim)
    title_model.load_state_dict(torch.load('temp/wikipedia_embeddings.pt', weights_only=True))

    ### Combine the features into a combined ML model
    # Num of days user has been around as first feature - ML predict output. Output is FloatTensor
    # user_days_raw = torch.load('temp/user_days_output.pt')
    # combimed_data = combine_feat(user_days_raw)
    # print(f"combimed_data shape: {combimed_data.shape}")
    # combined_ft_model = CombineFeats(500)

    # combined_out = combined_ft_model.forward(combimed_data)
    # print(f"combined_out shape: {combined_out.shape}")
    # pred_ft_model = HNUP()
    # pred_score = pred_ft_model.forward(combined_out)
    # print(f"Prediction score: {pred_score}")

if __name__ == "__main__":
    main()
