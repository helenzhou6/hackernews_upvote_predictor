from hackernewsupvotepredictor.predict_model_second_stage import reinstate_org_model
from hackernewsupvotepredictor.clean_data import title2emb, prep_features
import torch
import json
import pandas as pd
from datetime import datetime, timedelta

saved_w_pred_model = "temp/final_model.pt"

# elements necessary for using embeddings
with open("temp/vocabulary.json", "r") as fp:
    vocab = json.load(fp)

embs = torch.load('temp/wikipedia_embeddings.pt')            

# bes test set
def get_predicted_score(raw_data):
    dict_data = dict(
        title=[raw_data.title],
        user_created=[raw_data.user_created], 
        time=[raw_data.time],
    )
    df = pd.DataFrame(dict_data)
    print(df)

    df["user_created"] = pd.to_datetime(df["user_created"], unit="s")
    df["time"] = pd.to_datetime(df["time"], unit="s")
    _, features_b = prep_features(df)

    model = reinstate_org_model()
    model.load_state_dict(torch.load(saved_w_pred_model, weights_only=True))

    pred_score = model(features_b)
    return int(pred_score.detach().numpy())