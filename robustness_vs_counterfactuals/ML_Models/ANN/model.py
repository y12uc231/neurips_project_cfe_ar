import torch
from torch import nn

class ANN(nn.Module):
    def __init__(self, input_layer: int, hidden_layer: int = 100,
                 num_of_classes: int = 1, task: str = 'classification',
                 weighted_model: bool = True, train_set_size=None):
        super().__init__()
        
        # Layers
        self.input1 = nn.Linear(input_layer, hidden_layer)
        self.input2 = nn.Linear(hidden_layer, num_of_classes)
        self.task = task
        
        # Weight vectors
        if weighted_model:
            assert train_set_size is not None
            self.data_weights_vector = nn.Parameter(torch.ones(train_set_size), requires_grad=False)
        
        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.input1(x)
        output = self.relu(output)
        output = self.input2(output)
        if self.task == 'classification':
            output = self.sigmoid(output)
        return output
    
    def predict_with_logits(self, x: torch.tensor) -> torch.tensor:
        output = self.input1(x)
        output = self.relu(output)
        output = self.input2(output)
        return output
