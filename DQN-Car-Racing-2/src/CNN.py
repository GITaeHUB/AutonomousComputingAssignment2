import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(CNNActionValue, self).__init__()
        # Conv 레이어 정의
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 임시 입력을 통해 출력 크기 계산
        self._calculate_in_features()

        # 완전 연결 레이어 정의
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def _calculate_in_features(self):
        # 임시 텐서로 Conv 레이어의 출력 크기 계산
        with torch.no_grad():
            x = torch.zeros(1, 4, 84, 84)  # 입력 크기: [Batch, Channels, Height, Width]
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            self.in_features = x.numel()  # 전체 요소 수 계산

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.view((-1, self.in_features))  # 계산된 in_features로 변환
        x = self.fc1(x)
        x = self.fc2(x)
        return x
