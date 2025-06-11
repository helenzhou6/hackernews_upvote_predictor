from hackernewsupvotepredictor.download_cbow_rawdata import download_wiki_data
from hackernewsupvotepredictor.cbow import WordEmbeddings, train_cbow_model, Tokenizer
from hackernewsupvotepredictor.combined_feats import CombineFeats, combine_feat
from hackernewsupvotepredictor.predict_model import HNUP
import torch

def main():
    # set the dimension of embeddings
    embedding_dim = 64
    batch_size = 512

    # length of raw_data = 100,000,000
    # url = 'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8'
    # wiki_data = download_wiki_data(url)
    # truncated_wiki_data= wiki_data[:100000]

    with open("data/text8", "r") as f:
        wiki_data = f.read()

    # length of raw_data = 100,000,000
    wiki_data= wiki_data[:100000]

    tokenizer = Tokenizer(wiki_data)
    # Training Setup
    cbow_model = WordEmbeddings(wiki_data, tokenizer, embedding_dim, batch_size)
    # train_model(cbow_model, tokenizer, batch_size)

    # Num of days user has been around as first feature - ML predict output. Output is FloatTensor
    user_days_raw = torch.load('temp/user_days_output.pt')

    ### Combine the features into a combined ML model
    combimed_data = combine_feat(user_days_raw)
    print(f"combimed_data shape: {combimed_data.shape}")
    combined_ft_model = CombineFeats(500)

    combined_out = combined_ft_model.forward(combimed_data)
    print(f"combined_out shape: {combined_out.shape}")
    pred_ft_model = HNUP()
    pred_score = pred_ft_model.forward(combined_out)
    print(f"Prediction score: {pred_score}")

if __name__ == "__main__":
    main()
