import torch as tc
from torch import nn
from typing import Callable


class FFNN(nn.Module):
    '''Standard feed-forward neural network with the ability to easily add other modules like BatchNorm'''
    def __init__(self, architecture: list[int | nn.Module], nonlinearity: Callable[[], nn.Module] = nn.ReLU, *args, **kwargs):
        '''
        * architecture: List of integers and modules. Each pair of consecutive integers (ignoring modules) forms a linear layer that is placed at the spot of the latter integer. First entry must be an integer.
        * nonlinearity: Generating function for the nonlineraity that is placed after each linear layer except the last.
        '''
        super().__init__(*args, **kwargs)

        # enforce architecture constraint
        assert isinstance(architecture[0], int), "First architecture entry must be an integer! It defines the input size."

        # find sizes for linear layers
        layer_sizes = {i: n for i, n in enumerate(architecture) if isinstance(n, int)}
        i_last = max(layer_sizes.keys())
        layer_sizes_list = list(layer_sizes.items())
        layer_pairs = {i: (a, b) for (_, a), (i, b) in zip(layer_sizes_list[:-1], layer_sizes_list[1:])}

        # create module list
        modules: list[nn.Module] = []
        for i, a in enumerate(architecture[1:], start=1): # skip the first entry because it is always an integer (the respective linear layer is placed at the spot of the second integer)
            if isinstance(a, int):
                # add linear layer
                modules.append(nn.Linear(*layer_pairs[i]))

                # for each linear layer but the last, add nonlinearity
                if i != i_last:
                    modules.append(nonlinearity())
            else:
                # add custom layer
                modules.append(a)

        # create callable from module list
        self.layers = nn.Sequential(*modules)

        # remember parameters for the clone function
        self.architecture = architecture.copy()
        self.nonlinearity = nonlinearity


    def clone(self):
        '''Creates a deep copy of itself'''
        out = self.__class__(self.architecture, self.nonlinearity) # self.__class__ is used here to make it compatible with child classes
        out.load_state_dict(self.state_dict())
        return out
    

    def forward(self, X: tc.Tensor) -> tc.Tensor:
        return self.layers(X)
