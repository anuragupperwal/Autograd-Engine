import numpy as np

class SGD:
    ''' Standard Stochastic Gradient Descent optimizer. '''
    def __init__(self, params, lr=1e-3, reg=0) -> None:
        ''' 
        Instance of the SGD optimizer.
        
        @param params (list): list of all Parameter or Tensor (with requires_grad = True) to be optimized by Adam.
        params is usually set to nn.Module.parameters(), which automatically returns all parameters in a list form.
        @param lr (float): scalar multiplying each learning step, controls speed of learning.
        @param reg (float): scalar controling strength l2 regularization.
        '''
        self.params = params
        self.lr = lr
        self.reg = reg
        

    def step(self):
        ''' Updates all parameters in self.params. '''
        for param in self.params:
            param._data = param._data - (self.lr * param.grad) - (self.lr * self.reg * param._data)

    def zero_grad(self):
        ''' Sets all the gradients of self.params to zero. '''
        for param in self.params:
            param.zero_grad()
