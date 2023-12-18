import torch
import torch.nn as nn

class IdentityVerificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IdentityVerificationModel, self).__init__()

        # 완전 연결 레이어
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 순전파
        x = x.view(x.size(0), -1)  # 평탄화(flatten)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x