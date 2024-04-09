import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, model_a, model_b):
        super(CombinedModel, self).__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x):
        # Forward pass through Model A
        out_a = self.model_a(x)
        
        # Forward pass through Model B with the output of Model A
        out_b = self.model_b(out_a)

        return out_b

# Assuming you have Model A and Model B defined elsewhere
#model_a = YourModelA()
#model_b = YourModelB()

# Combine the two models into a single model
#combined_model = CombinedModel(model_a, model_b)

#In this example, CombinedModel takes two models, model_a and model_b, as arguments in its constructor. In the forward method, it first passes the input x through Model A (model_a) to get out_a, and then it passes out_a through Model B (model_b) to obtain the final output.

#You can then train and use combined_model just like any other PyTorch model.

#Make sure that the input and output shapes of Model A and Model B are compatible, and that the models have been properly trained or initialized before combining them.





