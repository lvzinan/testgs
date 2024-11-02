import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.resnet import resnet18

class optimise_cam(nn.Module):

    def __init__(self):
        super(optimise_cam, self).__init__()

        self.resnet = resnet18(pretrained=True,in_channels=[64,128,256,512])
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # # 第二层卷积，输入通道为16，输出通道为32
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # # 第三层卷积，输入通道为32，输出通道为64
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 全连接层
        self.fc1_cam = nn.Linear(64 * 225 * 400, 128)
        # 输出层，输出7个类别
        self.fc2_cam = nn.Linear(128, 24)

        self.fc1 = nn.Linear(36, 128)
        self.fc2 = nn.Linear(128,256)
        self.fc_r = nn.Linear(256, 9)
        self.fc_t = nn.Linear(256, 3)

    def forward(self, x, R, T):
        x = x.view(1,x.size(0),x.size(1),x.size(2))
        y = self.resnet(x)
        # 卷积层 + ReLU + 池化
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x = F.interpolate(x,[400, 800])
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x = F.interpolate(x, [200, 400])
        # x = F.relu(self.conv3(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = y[0]


        # 展平
        x = x.view(x.size(0), -1)  # 展平为(batch_size, 64*32*32)

        # 全连接层
        x = F.relu(self.fc1_cam(x))
        x = self.fc2_cam(x)
        R = torch.tensor(R).float().cuda()
        T = torch.tensor(T).float().cuda()
        R = R.contiguous().view(1, 9)
        T = T.contiguous().view(1, 3)
        x = torch.cat([x, R, T], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        R = self.fc_r(x)
        T = self.fc_t(x)
        return R,T