import torch
import torch.nn as nn
# from hackernewsupvotepredictor.predict_model2 import get_final_pred

class CombinedAllModel(nn.Module):
    def __init__(self, 
                 base_models, 
                 combined_model, 
                #  predict_model
                 ):
        super(CombinedAllModel, self).__init__()

        # self.init_model = nn.Linear(in_features=100, out_features=544)

        # base_models: list of models like [model1, model2, ...]
        self.base_models = nn.ModuleList(base_models)

        # Example: 1 linear layer per base model
        self.linears = nn.ModuleList([
            nn.Linear(1525, 1) for _ in base_models
        ])

        # Final processing models
        self.combined_model = combined_model  # takes all base outputs
        # self.predict_model = predict_model    # final prediction

    def forward(self, x):

        titles_b, users_b = x
        # titles_b = titles_b.long()
        # out = self.init_model(titles_b)
        # print("post-init-model")

        outputs = []
        base_models = self.base_models[0]
        out = base_models(titles_b) # this data should be token id
        linears = self.linears[0]
        out = linears(out)
        outputs.append(out)

        # for model, linear in zip(self.base_models, self.linears):
        #     print(f"title shape {titles_b.shape}")
        #     print(model)
        #     out = model(titles_b) # this data should be token id
        #     print(f"after one model {out.shape}")
        #     out = linear(out)
        #     outputs.append(out)

        # Combine outputs
        combined = self.combined_model(*outputs)

        # # Predict
        # result = torch.round(get_final_pred(combined))
        # # result = torch.round(self.predict_model(combined))
        return combined

