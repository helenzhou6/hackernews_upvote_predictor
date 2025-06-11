from hackernewsupvotepredictor.download_cbow_rawdata import download_wiki_data

def main():
    url = 'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8'
    wiki_data = download_wiki_data(url)

if __name__ == "__main__":
    main()
