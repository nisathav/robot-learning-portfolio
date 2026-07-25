short write-up (in your own words) of what autograd does and why zero_grad() matters.

Autograd: This helps to calculate the gradient of loss with respect to the associated parameters in the network. This helps us to understand the performance of neural network with respect to the changes in the input parameters. Using gradient decesent this learning tries to achieve the target values through multiple iterations of the dataset. 

zero_grad(): This helps to reset the data of the tensor while keeping it a valid tensor that future .backward() calls can continue accumulating into. This accumulation is a typical behaviour of the tensor which helps to keeps tack of multiple mini batches. 