import torch
import numpy as np

torch.manual_seed(42)

'''
The Autograd package in pytorch automatically calculates the gradients of the learnable parameters used
in the model. It is essential for optimization algorithms to update weights, and learn
'''

# set the requires_grad to true
x = torch.randn((3, ), requires_grad=True)

# this is like a simple "forward prop"
y = x + 2

# pytorch will now create a computational graph to calculate the gradient to calculate the backward pass
# this specific gradient function would be AddBackward, since the forward was adding
print(y)

z = torch.mul(torch.square(y), 2)
print(z)

z = z.mean()
print(z)

# to calculate the gradients, simply call .backward()
# print(x.grad) # will return None, as no gradients were calculated
z.backward() # dz/dx
print(x.grad) # this will return the gradients of z with respect to x

'''
the gradients are calculated through the chain rule, when multiple layers are involved
This is also called the vector jacobian method, which needs a vector to be dot multiplied

the vector is not needed for scalar gradients, but is when calculating the gradients of
values all at once.
'''

# clear the gradients from the first back prop
x.grad.zero_()

a = x + 2
b = torch.square(a) * 2
# a.backward() # doesn't work
# this v is like weights for the gradients. Similarly to the weighted sum for every layer
# these weights dicate how much each gradient matter to the specific gradient we want
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
b.backward(v)
print(x.grad)

'''
however, cost functions typically produce a single numerical metric, so this vector
jacobian product is not really needed
'''

'''
another important thing is to know how to prevent pytorch from tracking. 

for some calculations, we want it to be ommited from the gradient caluclation.
stuff such as loss values during predictions, where we don't even want a
computation graph for

frozen layers from transfer learning and fine-tuning also shouldn't be updated
parameters that shouldn't be learned

data processing operations shouldn't create a computational graph either.

Only quantities that affect loss values should be updated
'''

# method 1
# x.requires_grad_(False) or x.requires_grad = False # to not track gradients

# method 2
# x.detach() # tensor with no gradients

# method 3
# with torch.no_grad():
#  OPERATIONS GOES HERE AND NO GRADIENTS WILL BE CALCULATED FOR THEM

print("\n\n")

k = torch.randn((3, ), requires_grad=True)
print(k)

k.requires_grad_(False)
print(x)

l = k.detach()
print(l)

with torch.no_grad():
    o = x + 2
    print(o) # no grad_fn attribute

'''
NOTE: multiple grad calls on the same tensor will result in an accumulation of every gradient
It does NOT reset each time automatically
'''
