import pandas as pd

# NOTE: Need to uv add huggingface_hub

def download_hn_title_data():
    df = pd.read_parquet("hf://datasets/julien040/hacker-news-posts/story.parquet")
    df = df[df["score"] > 0]
    df_title = df["title"]
    df_titles = [item+" <ENDOFTITLE> " for item in df_title]
    concatenated_titles = "".join(item + " <ENDOFTITLE> " for item in df_titles)
    with open("data/hn_title_data", "w", encoding="utf-8") as f:
        f.write(concatenated_titles)

    