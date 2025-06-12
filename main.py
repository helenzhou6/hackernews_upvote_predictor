# from hackernewsupvotepredictor.download_cbow_rawdata import download_data_from_url
from hackernewsupvotepredictor.cbow import WordEmbeddings, train_cbow_model, Tokenizer
# from hackernewsupvotepredictor.combined_feats import CombineFeats, combine_feat
from hackernewsupvotepredictor.download_hn_title_data import download_hn_title_data
import torch.nn as nn
import torch

def main():
    # set the dimension of embeddings
    embedding_dim = 64
    batch_size = 512

    # Run download_cbow_rawdata.py to download
    with open("data/text8", "r") as f:
        wiki_data = f.read()
    wiki_data= wiki_data[:100000]

    # Training Setup
    wiki_tokenizer = Tokenizer(wiki_data, "temp/vocabulary.json")
    # cbow_model = WordEmbeddings(wiki_tokenizer, embedding_dim)
    # train_cbow_model(cbow_model, wiki_tokenizer, batch_size)

    # Loading the embeddings into title CBOW model
    # download_hn_title_data()  # to download the title data if not already there
    # with open("data/hn_title_data", "r") as f:
    #     hn_title_data = f.read()
    # hn_title_data = hn_title_data[:100000]
    title_model = WordEmbeddings(wiki_tokenizer, embedding_dim)
    title_model.load_state_dict(torch.load('temp/wikipedia_embeddings.pt', weights_only=True))

    ### Combine the features into a combined ML model
    # Num of days user has been around as first feature - ML predict output. Output is FloatTensor
    # user_days_raw = torch.load('temp/user_days_output.pt')
    # combimed_data = combine_feat(user_days_raw)
    # print(f"combimed_data shape: {combimed_data.shape}")
    # combined_ft_model = CombineFeats(500)

    # all_models(title_model, combined_ft_model, )

    # combined_out = combined_ft_model.forward(combimed_data)
    # print(f"combined_out shape: {combined_out.shape}")
    # pred_ft_model = HNUP()
    # pred_score = pred_ft_model.forward(combined_out)
    # print(f"Prediction score: {pred_score}")

if __name__ == "__main__":
    main()
