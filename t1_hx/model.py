
import torch
import torch.nn as nn
import torch.optim as optim

# TODO:
# instead of this class method, we could also use Torch.nn.Sequential to define the model.



class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=2):
        '''
        Very simple MLP with one layer.
        '''
        super(MLP, self).__init__()

        # Layer 1
        self.input_scaler = nn.BatchNorm1d(input_dim, affine=False)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.skip1 = nn.Linear(input_dim, hidden_dim, bias=False)
        # Layer 2
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)  # Will predict both temperatures

        # # Xavier initialisation
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    def forward(self, x):
        h1 = torch.relu(self.linear1(self.input_scaler(x)))
        h2 = self.linear2(h1)
        y = self.output(h2)
        return y

def empirical_loss(pred, target):
    return nn.MSELoss()(pred, target)


def train_model(model, train_data, train_targets, epochs=1000, 
                lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    # Fixed weights when not using meta-learning (PI-rMLP case)
    w = torch.tensor([100.0])
    
    for epoch in range(epochs):
        model.train() # Set model to training mode. 
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(train_data)
        # Compute losses
        loss = empirical_loss(pred, train_targets)
        # Backward pass for model parameters
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Total Loss={loss.item():.6f}, "
                  f"w={w.item():.6f}")

def test_model(model, test_data, test_targets):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        pred = model(test_data)
        loss = empirical_loss(pred, test_targets)
        print(f"Test Loss: {loss.item():.6f}")
        return pred