import torch
import torch.nn as nn
import hackernewsupvotepredictor

class CombinedAllModel(nn.Module):
    def __init__(self, base_models, combined_model, predict_model):
        super(CombinedAllModel, self).__init__()

        # base_models: list of models like [model1, model2, ...]
        self.base_models = nn.ModuleList(base_models)

        # Example: 1 linear layer per base model
        self.linears = nn.ModuleList([
            nn.Linear(16 * 5 * 5, 120) for _ in base_models
        ])

        # Final processing models
        self.combined_model = combined_model  # takes all base outputs
        self.predict_model = predict_model    # final prediction

    def forward(self, x):
        outputs = []

        for model, linear in zip(self.base_models, self.linears):
            out = model(x)
            out = linear(out)
            outputs.append(out)

        # Combine outputs
        combined = self.combined_model(*outputs)

        # Predict
        result = self.predict_model(combined)
        return result

