import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# === データ前処理（データ拡張あり） ===
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
transform_test = transforms.ToTensor()

training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform_train)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform_test)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# === デバイス設定 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# === ResNet50 の構築（1ch対応 & 10クラス出力） ===
model = resnet50(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(2048, 10)
model = model.to(device)

# === 損失関数・最適化手法・スケジューラ ===
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# === 訓練関数 ===
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# === 評価関数 ===
def evaluate(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    accuracy = correct / size
    return test_loss, accuracy

# === 学習ループ ===
test_losses = []
accuracies = []
best_loss = float('inf')
patience = 10
counter = 0

for epoch in range(200):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test_loss, accuracy = evaluate(test_dataloader, model)

    test_losses.append(test_loss)
    accuracies.append(accuracy)

    print(f"Test Accuracy: {accuracy:.2%}, Loss: {test_loss:.4f}")

    if test_loss < best_loss:
        best_loss = test_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("→ New best model saved!")
    else:
        counter += 1
        print(f"→ No improvement. Patience: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered!")
            break
    scheduler.step(test_loss)

# === グラフ保存 ===
os.makedirs("graph", exist_ok=True)
epochs = range(len(test_losses))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(epochs, test_losses, label='Test Loss', color='tab:blue')
ax1.set_title('Test Loss per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epochs, accuracies, label='Test Accuracy', color='tab:green')
ax2.set_title('Test Accuracy per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.tight_layout()
plt.savefig("graph/resnet50_metrics.pdf")
plt.show()

# === 混同行列の作成・保存 ===
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax_cm)
ax_cm.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("graph/resnet50_confusion_matrix.pdf")
plt.show()
