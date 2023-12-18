import torch
import torch.nn as nn
import torch.optim as optim

class AMLModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AMLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 모델 초기화
input_size = 10
hidden_size = 128
output_size = 1  # 이진 분류이므로 1
model = AMLModel(input_size, hidden_size, output_size)

# 손실 함수와 옵티마이저 정의
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 데이터 준비
# TODO: 학습 데이터를 로드하거나 생성하는 부분을 추가해야 합니다.
train_dataset = None
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')
