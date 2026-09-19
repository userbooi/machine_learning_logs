import torch
import numpy as np

torch.manual_seed(42)

'''
this is how to create an empty tensor
'''
x1 = torch.empty((1, )) # essentially a scalar
x2 = torch.empty((3, )) # 1d tensor
x3 = torch.empty((3, 4)) # 2d tensor
# this is different from torch.zeros, where it is explicitly set to 0
# torch.empty creates uninitialized values

# print(x1); print(x2); print(x3)

'''
random tensors
'''
x1 = torch.rand((3, 2))
# print(x1)

'''
numpy initializations also works
'''
x1 = torch.zeros((4, 2))
x2 = torch.ones(2, 1)

# print(x1); print(x2)

'''
check the datatype of a tensor
'''
# print(x1.dtype)

'''
change the datatype
'''
x1 = torch.zeros((2, 2), dtype=torch.int32)
x2 = x2.to(torch.double)
# print(x1, x1.dtype)
# print(x2, x2.dtype)

'''
check the shape of the tensor
'''
# print(x1.shape)
# print(x1.size()) # same thing as .shape

'''
construct tensor from data
'''
x1 = torch.tensor([2, 3, 1.201, 2.1, 9])
# print(x1, x1.dtype)

'''
basic operations with tensors
'''
x1 = torch.rand((2,2))
y1 = torch.rand((2,2))
z1 = x1 + y1
# z1 = torch.add(x1, y1)

# print(z1, end="\n\n")

# testing broadcasting
y2 = torch.rand((2, 1))
z2 = x1 + y2

# print(x1); print(y2, end="\n\n")
# print(z2)

'''
in place addition replaces the target with the result of the operation
'''
# all in place functions are followed by a _
# print(x1); print(y1)
# x1.add_(y1)
# print(x1)

'''these all extend to the other basic element-wise operations'''

'''
slicing operations (like numpy arrays)
'''
x1 = torch.rand(5, 3)

# print(x1, end="\n\n")
# print(x1[:, 1:2]) # keeps the dimensions
# print(x1[:, 1])   # drops the columns
# print(x1[2:4, 0:2])

'''
get the actual, scalar value from a tensor
ONLY USABLE FOR A TENSOR WITH 1 ELEMENT
'''
# print(x1[3, 1].item())

'''
reshape a tensor
'''
y1 = x1.view((15, 1))

# print(x1)
# print(y1)

'''
converting from numpy to torch and vice versa
'''

# these two share the same memory
# b is using np data structure to process the same information
a = torch.ones((5, ))
b = a.numpy()

# print(b, type(b), end='\n\n')
#
# # if any were modified in-place, both would be changed
# a.add_(torch.rand((5, )))
# print(b)
# b += 1
# print(a, end='\n\n')

a = np.ones((5, ))
b = torch.from_numpy(a)

# print(b)

'''
tensors are used in models, and the autograd function will calculate the gradients for 
back prop.

set the requires_grad=true

This is necessary for learnable parameters that will be updated in back prop
'''
x1 = torch.ones((5, ), requires_grad=True, dtype=torch.int32)

# print(x1)
