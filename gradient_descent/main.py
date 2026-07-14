import numpy as np
import matplotlib.pyplot as plt

'''
gradients are vector values made up of its partial derivatives.
the gradient vector point in the direction of steepest ascent

1. A function has 1 independent variable:
    - in this case, the gradient is the derivative of the function with respect to the independent variable
    - ex. y = 3x^2 + 2x - 4
          the gradient would be:
          y = 6x + 2
2. A function has multiple variables:
    - in this case, the gradient is based on all of its variables
    - ex. f(x, y) = x^2 + 3xy
          the gradient would be:
          grad_f = <grad_f_x, grad_f_y>
    - find the Partial Derivative, where every other variable is treated as a constant
    - ex. f(x, y) = x^2 + 3xy
          the gradient in terms of x would be:
          grad_f_x = 2x + 3y
          the gradient in terms of y would be:
          grad_f_y = 3x
          
          in this case, the gradient would be:
          grad_f = <2x+3y, 3x>

gradient descent uses gradients to update parameters so that it can achieve a minimum loss value.

for example, when there is parameters weight (w) and bias (b), gradient descent update rule is:

w = w - a(grad_l_w)
b = b - a(grad_l_b)

where: w is weight
       b is bias
       a is learning rate
       l is loss

'''

#============================== Mean Square Error ===============================
np.random.seed(42)
n = 200

X = np.linspace(-2, 3, n).reshape(-1, 1)
# generates a target array of scattered points that should result in a line of best fit
# of y = 3x - 1.25
y = 3 * X.ravel() - 1.25 + 0.7 * np.random.randn(n)

def gradient_descent_MSE(X, y, loss, w=0.0, b=0.0, a=0.003, epoch=1000):
    '''
    the formula for MSE is: (the extra 2 in the division is used to reduce the 2 can results from differentiation)
    L = 1/2n * sigma((y_pred - y)^2)

    for linear regression where y_pred = wx + b,

    grad_L_w = 1/2n * sigma(2(y_pred - y)(x))
             = 1/n * sigma((y_pred-y)x)

    grad_L_b = 1/2n * sigma(2(y_pred - y)(1))
             = 1/n * sigma(y_pred-y)
    '''

    for _ in range(epoch):
        y_pred = w * X.ravel() + b
        errors = y_pred - y
        curr_loss = (errors**2).mean() / 2
        loss.append(curr_loss)
        # can use the mean() method to calculate
        dw = (errors * X.ravel()).mean()
        db = errors.mean()
        w -= a * dw
        b -= a * db

    return w, b

#================================== testing ==================================
loss = []
w, b = gradient_descent_MSE(X, y, loss)

for l in loss:
    print(l)

print(w, b)

#================================== plot =====================================
plt.figure()
plt.plot(loss, color="green")
plt.title("Linear Regression — Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(X, y, color="lightgreen")
idx = np.argsort(X.ravel())
plt.plot(X[idx], (w * X + b)[idx], color="green", linewidth=2)
plt.title("Linear Regression — Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
