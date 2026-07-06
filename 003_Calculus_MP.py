import numpy as np

# training data
x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

# starting values
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # prediction
    y_pred = w * x + b

    # loss
    loss = np.mean((y_pred - y) ** 2)

    # gradients
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))

    # update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss, "w:", w, "b:", b)

print("Final weight:", w)
print("Final bias:", b)