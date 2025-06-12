import torch
import torch.nn as nn

class CombinedAllModel(nn.Module):
    def __init__(self, base_models, combined_model, predict_model):
        super(CombinedAllModel, self).__init__()

        # base_models: list of models like [model1, model2, ...]
        self.base_models = nn.ModuleList(base_models)

        # Example: 1 linear layer per base model
        self.linears = nn.ModuleList([
            nn.Linear(100, 1) for _ in base_models
        ])

        # Final processing models
        self.combined_model = combined_model  # takes all base outputs
        self.predict_model = predict_model    # final prediction

    def forward(self, x):

        titles_b, users_b = x

        outputs = []

        for model, linear in zip(self.base_models, self.linears):
            print(f"title shape {titles_b.shape}")
            out = model(titles_b) # this data should be token id
            print(f"after one model {out.shape}")
            out = linear(out)
            outputs.append(out)

        # Combine outputs
        combined = self.combined_model(*outputs)

        # Predict
        result = self.predict_model(combined)
        return result

