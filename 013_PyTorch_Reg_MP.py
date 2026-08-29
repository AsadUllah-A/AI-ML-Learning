import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, random_split

# -----------------------------
# 1. Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2. Data
# -----------------------------
X = torch.tensor([
    [0.9, 0.8, 0.9],
    [0.7, 0.6, 0.7],
    [0.5, 0.5, 0.5],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.2],
    [0.8, 0.9, 0.85],
    [0.95, 0.9, 0.98],
    [0.4, 0.45, 0.4],
    [0.6, 0.55, 0.6],
    [0.15, 0.25, 0.2],
    [0.75, 0.7, 0.8],
    [0.35, 0.3, 0.35]
], dtype=torch.float32)

# Target: Marks
y = torch.tensor([
    [95.0],
    [82.0],
    [70.0],
    [55.0],
    [45.0],
    [90.0],
    [98.0],
    [60.0],
    [75.0],
    [40.0],
    [85.0],
    [50.0]
], dtype=torch.float32)

# -----------------------------
# 3. Dataset Split
# -----------------------------
dataset = TensorDataset(X, y)

train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)

# -----------------------------
# 4. Model
# -----------------------------
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(3, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


model = RegressionModel().to(device)

# -----------------------------
# 5. Loss and Optimizer
# -----------------------------
loss_function = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

# -----------------------------
# 6. Helper Function for MAE
# -----------------------------
def regression_mae(predictions, y_true):
    mae = torch.abs(predictions - y_true).mean()
    return mae.item()

# -----------------------------
# 7. Training + Validation Loop
# -----------------------------
epochs = 100

for epoch in range(epochs):

    # Training
    model.train()

    train_loss = 0
    train_mae = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(X_batch)

        loss = loss_function(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mae += regression_mae(predictions, y_batch)

    train_loss = train_loss / len(train_loader)
    train_mae = train_mae / len(train_loader)

    # Validation
    model.eval()

    val_loss = 0
    val_mae = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            loss = loss_function(predictions, y_batch)

            val_loss += loss.item()
            val_mae += regression_mae(predictions, y_batch)

    val_loss = val_loss / len(val_loader)
    val_mae = val_mae / len(val_loader)

    if (epoch + 1) % 10 == 0:

        print(
            "Epoch:", epoch + 1,
            "| Train Loss:", round(train_loss, 4),
            "| Train MAE:", round(train_mae, 4),
            "| Val Loss:", round(val_loss, 4),
            "| Val MAE:", round(val_mae, 4)
        )