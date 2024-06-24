import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super(TCNBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(output_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
    

class TCN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, list_of_dim_of_layers: list, 
                 hidden_size: int = 64, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super(TCN, self).__init__()
        self.list_of_dim_of_layers = [input_channels] + list_of_dim_of_layers + [hidden_size]
        self.layers = nn.ModuleList()
        for i in range(len(self.list_of_dim_of_layers) - 1):
            dilation = 2 ** i
            conv = nn.Sequential(TCNBlock(input_channels=self.list_of_dim_of_layers[i], output_channels=self.list_of_dim_of_layers[i + 1],
                                          kernel_size=kernel_size, dilation=dilation),
                                 nn.Dropout(dropout))
            self.layers.append(conv)
        self.fc = nn.Linear(hidden_size, output_channels)
    
    def forward(self, x):
        batch_size, input_size, T = x.size()
        for layer in self.layers:
            x = layer(x)
        x = F.avg_pool1d(x, T).squeeze(-1)
        x = self.fc(x)
        return x