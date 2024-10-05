import torch
import torch.nn as nn
from typing import Tuple, List



class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_xh = nn.Parameter(torch.zeros(hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            input @ self.W_xh.t() + self.bias_xh +
            hidden @ self.W_hh.t() + self.bias_hh
        )


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList([
            RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, input: torch.Tensor, hidden: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len, batch_size, _ = input.size() # seq_len, btz, input_size 

        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input.device)
        
        output = torch.zeros(seq_len, batch_size, self.hidden_size, device=input.device)

        for t in range(seq_len):
            layer_input = input[t]
            for l, cell in enumerate(self.cells):
                hidden_l = hidden[l]
                hidden_l = cell(layer_input, hidden_l)
                layer_input = hidden_l
                hidden[l] = hidden_l
            output[t] = layer_input
            
        return output, hidden