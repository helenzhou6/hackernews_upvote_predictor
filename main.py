from hackernewsupvotepredictor.download_cbow_rawdata import download_wiki_data
from hackernewsupvotepredictor.cbow import WordEmbeddings, train_model, Tokenizer

def main():
    # length of raw_data = 100,000,000
    # url = 'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8'
    # wiki_data = download_wiki_data(url)
    # truncated_wiki_data= wiki_data[:100000]
    with open("data/text8", "r") as f:
        raw_data = f.read()

    # length of raw_data = 100,000,000
    raw_data= raw_data[:100000]

    tokenizer = Tokenizer(raw_data)
    # Training Setup
    model = WordEmbeddings(raw_data, tokenizer)
    train_model(model, tokenizer)

if __name__ == "__main__":
    main()
