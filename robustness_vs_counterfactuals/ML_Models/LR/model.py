import torch
import torch.nn as nn

class Regression(nn.Module):
    def __init__(self, input_dim: int, num_of_classes: int = 1, task: str = 'classification',
                 weighted_model: bool = True, train_set_size=None):
        
        super().__init__()
        
        # Auxiliary information
        self.input_dim = input_dim
        self.num_of_classes = num_of_classes
        self.task = task
        self.train_set_size = train_set_size
        
        # Weight vectors
        if weighted_model:
            assert self.train_set_size is not None
            self.data_weights_vector = nn.Parameter(torch.ones(self.train_set_size), requires_grad=False)

        # Layers
        self.linear = nn.Linear(self.input_dim, self.num_of_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.task == 'classification':
            return self.sigmoid(self.linear(x))
        else:
            return self.linear(x)
    
    def predict_with_logits(self, x: torch.tensor) -> torch.tensor:
        output = self.linear(x)
        return output
