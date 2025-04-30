import torch
from torch import nn
import torch.optim.adamw
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F  # 活性化関数にF.reluを使う場合に必要
from torch.optim.lr_scheduler import ReduceLROnPlateau


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])


transform_test = transforms.ToTensor()

# 訓練データをdatasetsからダウンロード
training_data = datasets.FashionMNIST(
    root="data",                                # データの保存先
    train=True,                                 # 訓練データとする(今回は60000枚)
    download=True,                              # データを(ネットから)ダウンロード
    transform=transform_train                   # データの前処理（データ拡張）
)

# テストデータをdatasetsからダウンロード
test_data = datasets.FashionMNIST(
    root="data",
    train=False,                                # テストデータとする(↑とは違う10000枚)
    download=True,
    transform=transform_test
)

batch_size = 64             # バッチサイズを指定

# データローダーの作成(データによってはこれの前にデータの前処理が必要)
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)                 # N:バッチサイズ, C:チャンネル数, H:高さ, W:幅
    print("Shape of y: ", y.shape, y.dtype)                     # yはラベル(バッチと同じ数)
    break

# 訓練に際して、可能であればGPU（cuda）を設定。GPUが搭載されていない場合はCPUを使用
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#CNNモデルの定義
class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()                  #親クラスの初期化
        self.conv_stack = nn.Sequential(                    #畳み込み+プーリング層（抽出）の定義
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     #入力:1ch → 出力:32ch、一枚の画像を画像を32通りの視点で見る，カーネル（フィルター）サイズ3x3，出力サイズを保つためにpadding=1
            nn.ReLU(),                                      #活性化関数（ReLU）を適用
            nn.MaxPool2d(2),                                # 28x28 → 14x14　画像圧縮　プーリングを2x2に分けて最大値を残す

            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 出力:64chに，情報をより抽象化する
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 14x14 → 7x7
        )

        self.flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.fc_stack(x)
        return x
    

model = CNNNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()                                     #損失関数（ロス関数）の定義、これを使ってモデルの誤差を計算
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)            #最適化手法の設定（今回は確率的勾配法、SGD）、学習率（lr）はパラメータ更新の幅、学習率を自動調整するスケジューラというものもある
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)  # 学習率を自動調整するスケジューラの設定（学習率を減少させる条件を指定）

# 学習を行う関数を定義
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)                                  #データセットの総データ数を取得
    for batch, (X, y) in enumerate(dataloader):                     #1バッチずつデータローダーからデータを取得、バッチのループ回数も取得
        X, y = X.to(device), y.to(device)                           #データをGPUに移動（モデルとデータが同じデバイスにないといけない）
        
        # 損失誤差を計算
        pred = model(X)                                             #モデルにデータを入力して予測値を取得
        loss = loss_fn(pred, y)                                     #予測値と実際のラベルを比較して損失誤差を計算
        
        # バックプロパゲーション
        optimizer.zero_grad()                                       #勾配を初期化（前のバッチの勾配が残っていると誤差が大きくなる）
        loss.backward()                                             #誤差を逆伝播させて勾配（どの方向に動かしたら損失が減るのか）を自動で計算
        optimizer.step()                                            #計算した勾配を使ってパラメータを更新（学習）

        if batch % 100 == 0:                                        #100バッチごとに進捗を表示（％は余りの計算）
            loss, current = loss.item(), batch * len(X)             #損失誤差を数値に変換して、現在の処理枚数（バッチの数×バッチのデータ量）を計算
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  #進捗を表示

def test(dataloader, model):
    size = len(dataloader.dataset)                                                          #データセットの総データ数を取得
    model.eval()                                                                            #モデルを推論モードに変更（ドロップアウトなし、バッチ正規化は学習で得た値に固定と、挙動が変わる）
    test_loss, correct = 0, 0                                                               #１エポック全体の損失誤差と正解数を初期化
    with torch.no_grad():                                                                   #勾配計算を行わない（推論時は必要ない（勾配を使ったパラメータ変化はない）ので、計算量を減らすために無効化）             
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)                                               #データをモデルと同じGPUに移動
            pred = model(X)                                                                 #モデルにデータを入力して予測値（ロジット）を取得
            test_loss += loss_fn(pred, y).item()                                            #損失誤差を計算して加算
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()                 #正解数を計算（argmaxで最大値のインデックスを取得して、実際のラベルと比較して正解数をカウント）
    test_loss /= size                                                                       #全体の損失誤差をデータ数で割って平均値を計算
    correct /= size                                                                         #全体の正解数をデータ数で割って平均値を計算             
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") #進捗を表示


# テスト結果記録用リストを追加
test_losses = []
accuracies = []
best_loss = float('inf')
patience = 10
counter = 0

epochs = 200  # エポック数（学習回数）を指定、増やしすぎても頭打ちになる
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")  # エポック数を表示
    train(train_dataloader, model, loss_fn, optimizer)

    # ====== test()関数の代わりに、評価結果を記録 ======
    model.eval()
    size = len(test_dataloader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    accuracy = correct / size

    test_losses.append(test_loss)
    accuracies.append(accuracy)

    print(f"Test Accuracy: {accuracy:.2%}, Loss: {test_loss:.4f}")

# === EarlyStopping判定 ===
    if test_loss < best_loss:
        best_loss = test_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")  # モデルを保存！
        print("→ New best model saved!")
    else:
        counter += 1
        print(f"→ No improvement. Patience: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered!")
            break

# --- 学習率スケジューラ処理 ---
scheduler.step(test_loss)   # test_lossに基づいて調整

print("Training done.")

print("Done!")  # 1エポック学習完了のメッセージを表示

# ====== グラフ描画 ======
epochs = range(len(test_losses))  # x軸：エポック番号

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1行2列のレイアウト

# --- 左側：Loss ---
ax1.plot(epochs, test_losses, label='Test Loss', color='tab:blue')
ax1.set_title('Test Loss per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# --- 右側：Accuracy ---
ax2.plot(epochs, accuracies, label='Test Accuracy', color='tab:green')
ax2.set_title('Test Accuracy per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

# ====== グラフをPDF保存 ======
plt.tight_layout()
plt.savefig("graph/ver9-1.pdf")  # グラフをPDF保存
plt.show()

# ===== 混同行列の表示 =====
classes = [                                                             # クラス名を人にわかるように定義
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()  # 推論モード

y_true = []
y_pred = []

with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        predicted = pred.argmax(1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# ====== 混同行列の表示とPDF保存 ======
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))  # 混同行列用のFigureを明示的に作る
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax_cm)
ax_cm.set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("graph/ver8-2.pdf")  # 混同行列をPDF保存
plt.show()

torch.save(model.state_dict(), "model_ver.8.pth")             # 学習したモデルのパラメータを保存
print("Saved PyTorch Model State to model_ver.8.pth")         # モデルの保存が完了したメッセージを表示
#model = CNNNetwork()                                          # 新しいモデル（空）を作る
#model.load_state_dict(torch.load("model_ver.8.pth"))          # 学習したモデルのパラメータを読み込む

model.eval()                                                            # モデルを推論モードに変更（ドロップアウトなし、バッチ正規化は学習で得た値に固定）
x, y = test_data[0][0], test_data[0][1]                                 # テストデータの最初の1枚とその答えを取得
with torch.no_grad():                                                   # 勾配計算を行わない（推論時は必要ない（勾配を使ったパラメータ変化はさせない）ので、計算量を減らすために無効化）
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]          # 予測値と実際のラベルを取得
    print(f'Predicted: "{predicted}", Actual: "{actual}"')              # 予測値と実際のラベルを比較


#model_ver.1:エポック５→10、最後のnn.ReLu()を削除,Test Accuracy: 70.70%, Loss: 0.0124
#model_ver.2:エポック10→20
#model_ver.3:エポック20→200
#model_ver.4:学習率を5e-4,層を４層に,最適化手法をAdamに変更
#model_ver.5:学習率を1e-4,
#model_ver.6:MLP→CNNに変更
#model_ver.7:アーリーストッピング導入、
#model_ver.8:Adam→AdamWに変更，オートスケジューラ・データ拡張導入，
