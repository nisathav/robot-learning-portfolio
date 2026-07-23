import torch

torch.manual_seed(42)  # reproducibility  non-negotiable per the task doc

# True relationship: y = 3x + 5 (plus a little noise)
x = torch.linspace(0, 10, 100)
y_true = 3 * x + 5 + torch.randn(100) * 0.5

w = torch.tensor(2.0, requires_grad=True)   # start w somewhere random/arbitrary, NOT at 3.0
b = torch.tensor(4.0, requires_grad=True)   # start b somewhere random/arbitrary, NOT at 5.0

lr = 0.01  # <- you'll need to comment on why this value, or note it's a guess to tune #keeping it minimum for now

for epoch in range(1000):
    # 1. Forward pass
    y_pred = w * x + b

    # 2. Loss (MSE)
    loss = ((y_pred - y_true)**2).mean()

    # 3. Backward
    loss.backward()

    # 4. Update — IMPORTANT: must be wrapped in `with torch.no_grad():`
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 5. Zero the gradients
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}, w={w.item():.3f}, b={b.item():.3f}")