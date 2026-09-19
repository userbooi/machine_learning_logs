import torch


# for the linear function y = 2*x
x = torch.tensor(1.0) # no requires_grad because values will not be updated
y = torch.tensor(2.0)

w = torch.tensor(0, dtype=torch.float32, requires_grad=True)

y_hat = x * w
s = y_hat - y

# squared loss - only one example, so no mean
# recall:
#   - cost is for many examples
#   - loss is for one example
loss = torch.square(s)
loss.backward()

print(w.grad)

### update the weights w -= lr * w.grad

