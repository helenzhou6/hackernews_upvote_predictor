# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class CombinedAllModel(nn.Module):
#     def __init__(self):
#         super(TheModelClass, self, model1, model2, model3, model4).__init__()
#         self.basemodel1 = model1 # cbow1 free weights
#         self.basemodel2 = model2 # cbow2 free weights
#         self.basemodel3 = model3
#         self.basemodel4 = model4
#         self.lin1 = nn.Linear(16 * 5 * 5, 120)

#     def forward(self, x):
#         x = self.basemodel1(x) # cbow1
#         x1 = self.lin1(x)

#         x = self.basemodel2(x) # cbow2
#         x2 = self.lin2(x)

#         x = self.basemodel3(x1, x2) # combined_model
#         x = self.basemodel4(x) # predict_model
#         return x

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

        titles_b, users_b = x

        outputs = []

        for model, linear in zip(self.base_models, self.linears):
            out = model(titles_b) # this data should be 
            out = linear(out)
            outputs.append(out)

        # Combine outputs
        combined = self.combined_model(*outputs)

        # Predict
        result = self.predict_model(combined)
        return result

# Initialize model
model1 = TheModelClass() # cbow model1
model2 = TheModelClass() # cbow model1
model3 = TheModelClass() # cbow model1

model1.train()
model2.train()
model3.train()

model1.save()
model2.save()
model3.save()

model4 = CombineFeats() # combined features model
model5 = FinalPred() # final score predicting model

# Step 1: Initialize model
combined_model = CombinedAllModel(
    base_models=[model1, model2, model3],
    combined_model=model4,
    predict_model=model5
)


# optimizer
# loss_func


### during training

# Step 2: Freeze base models
for param in combined_model.base_models.parameters():
    param.requires_grad = False

# Step 3: Define optimizer (important: after freezing)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, combined_model.parameters()),
    lr=1e-3
)


