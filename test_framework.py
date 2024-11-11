import neuralforge as forge
import neuralforge.nn as nn
import unittest
import numpy as np
import os
import matplotlib.pyplot as plt
import time


from neuralforge.nn.optim import SGD
from neuralforge.utils import plot_gradient_flow

class TestNeuralForge(unittest.TestCase):
    '''This class tests the functionalities of the framework in three levels of complexity.'''

    def test_autograd(self):
        '''
        This function tests whether the loss converges to zero in a spelled-out forward
        propagation, with weights explicitly declared.
        '''

        start_time = time.time()


        # Define loss function as Cross Entropy Loss:
        loss_func = nn.CrossEntropyLoss()

        # Instantiate input and output:
        x = forge.randn((8,4,5))
        y = np.random.randint(0,50,(8,4))

        # Instantiate Neural Network's Layers:
        w1 = forge.tensor(np.random.randn(5,128) / np.sqrt(5), requires_grad=True) 
        relu1 = nn.ReLU()
        w2 = forge.tensor(np.random.randn(128,128) / np.sqrt(128), requires_grad=True)
        relu2 = nn.ReLU()
        w3 = forge.tensor(np.random.randn(128,50) / np.sqrt(128), requires_grad=True)


        #to Visualize the training loss
        losses = []  # Track losses
        
        # Instantiate the optimizer with model parameters
        params = [w1, w2, w3]  # Replace with your model's parameters
        optimizer = SGD(params, lr=0.005)

        # Initialize gradient tracker
        grad_magnitudes = {"w1": [], "w2": [], "w3": []}
        iterations = []

        # Training Loop:
        for _ in range(2500):
            #forward pass
            z = x @ w1
            z = relu1(z)
            z = z @ w2
            z = relu2(z)
            z = z @ w3
            
            # Get loss:
            loss = loss_func(z, y)

            # Backpropagate the loss using neuralforge.tensor:
            loss.backward()


            # Track gradient magnitudes
            grad_magnitudes["w1"].append(np.linalg.norm(w1.grad))
            grad_magnitudes["w2"].append(np.linalg.norm(w2.grad))
            grad_magnitudes["w3"].append(np.linalg.norm(w3.grad))
            iterations.append(_)


            # Update the weights:
            # w1 = w1 - (w1.grad * 0.005) 
            # w2 = w2 - (w2.grad * 0.005) 
            # w3 = w3 - (w3.grad * 0.005) 
            #instead of manually calculating weights using optimiser here
            optimizer.step()

            # Reset the gradients to zero after each training step:
            loss.zero_grad_tree()

            # Append current loss for visualization
            losses.append(loss._data.item())
            if(_%200 == 0):
                print(f"Iteration {_}, Loss: {loss._data.item()}")


        assert loss._data < 3e-1, "Error: Loss is not converging to zero in autograd test."
        # Plot the training loss over iterations
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Iterations")
        plt.show()

        # Plot gradient flow after training
        plot_gradient_flow(grad_magnitudes, iterations)


        end_time = time.time()
        print(f"Execution Time for Autograd Engine: {end_time - start_time:.2f} seconds")


    def test_with_pytorch(self):
        start_time = time.time()

        import torch  # Ensure PyTorch is imported locally in this function to avoid global conflicts
        import torch.nn as nn
        import torch.optim as optim
        # Define data
        x = torch.randn(8, 5, requires_grad=True)
        y = torch.randint(0, 50, (8,), dtype=torch.long)

        # Define model
        model = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.005)

        losses = []
        for i in range(1000):
            # Forward pass
            z = model(x)

            # Loss computation
            loss = loss_func(z, y)
            losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Print loss periodically
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

        # Plot loss curve
        plt.plot(losses, label="PyTorch Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("PyTorch Training Loss")
        plt.legend()
        plt.show()

        end_time = time.time()
        print(f"Execution Time for Pytorch: {end_time - start_time:.2f} seconds")




if __name__ == '__main__':
    unittest.main()