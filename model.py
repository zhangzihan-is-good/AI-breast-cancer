from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.best_loss = 1000000
        self.best_accuracy = 0

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device


    def register_vars(self,**kwargs):
        if kwargs.get('task') == 'fine_tune':
            self.dropout_rates = {'input': 0, 'green': 0.35,'Up_green': 0}
        else:
            self.dropout_rates = {'input': 0, 'green': 0.2, 'Up_green': 0.2}

        self.inChannels = 1
        self.outChannels = 1
        self.model_depth = 2
        self.use_cuda = kwargs.get('cuda')

    def load_partial_state_dict(self, state_dict, load_encoder=True, load_decoder=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if load_encoder and name.startswith('encoder.'):
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            if load_decoder and name.startswith('decoder.'):
                if name not in own_state:
                    continue
                own_state[name].copy_(param)


#Encoder
class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.4):
        super(GreenBlock, self).__init__()
        self.Drop_Rate = drop_rate
        # Define block
        self.block = nn.Sequential(OrderedDict([
            ('group_norm0', nn.GroupNorm(num_channels=in_channels, num_groups=1)),
            ('relu0', nn.LeakyReLU(inplace=True)),
            ('conv0', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('group_norm1', nn.GroupNorm(num_channels=out_channels, num_groups=1)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ]))

    def forward(self, inputs):
        x_res = inputs
        x = torch.nn.functional.dropout(self.block(inputs), p=self.Drop_Rate, training=self.training)
        return x + x_res
  
class Encoder(BaseModel):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.register_vars(**kwargs)
        self.down_block1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(self.inChannels, self.model_depth, kernel_size=3, stride=1, padding=1)),
            ('sp_drop0', nn.Dropout2d(self.dropout_rates['input'])),
            ('green0', GreenBlock(self.model_depth, self.model_depth, self.dropout_rates['green'])),
            ('downsize_0', nn.Conv2d(self.model_depth, self.model_depth * 2, kernel_size=3, stride=2, padding=1))]))
        self.down_block2 = nn.Sequential(OrderedDict([
            ('green10', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('green11', GreenBlock(self.model_depth * 2, self.model_depth * 2, self.dropout_rates['green'])),
            ('downsize_1', nn.Conv2d(self.model_depth * 2, self.model_depth * 4, kernel_size=3, stride=2, padding=1))]))
        self.down_block3 = nn.Sequential(OrderedDict([
            ('green20', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('green21', GreenBlock(self.model_depth * 4, self.model_depth * 4, self.dropout_rates['green'])),
            ('downsize_2', nn.Conv2d(self.model_depth * 4, self.model_depth * 8, kernel_size=3, stride=2, padding=1))]))
        self.final_block = nn.Sequential(OrderedDict([
            ('green30', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green31', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green32', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green'])),
            ('green33', GreenBlock(self.model_depth * 8, self.model_depth * 8, self.dropout_rates['green']))]))

    def forward(self, x):
        x = self.down_block1(x)
        x = self.down_block2(x)
        x = self.down_block3(x)
        x = self.final_block(x)
        return x

#Decoder
class UpGreenBlock(nn.Sequential):
    def __init__(self, in_features, out_features, Drop_Rate):
        super(UpGreenBlock, self).__init__()
        # Use Conv2d for 2D data
        self.add_module('conv', nn.ConvTranspose2d(in_features, out_features, kernel_size=4, stride=2, padding=1))
        self.add_module('green', GreenBlock(out_features, out_features, Drop_Rate))

class Decoder(BaseModel):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.register_vars(**kwargs)
        self.upgreen0 = UpGreenBlock(self.model_depth * 8, self.model_depth * 4, self.dropout_rates['Up_green'])
        self.upgreen1 = UpGreenBlock(self.model_depth * 4, self.model_depth * 2, self.dropout_rates['Up_green'])
        self.upgreen2 = UpGreenBlock(self.model_depth * 2, self.model_depth, self.dropout_rates['Up_green'])
        self.blue_block = nn.Conv2d(self.model_depth, self.model_depth, kernel_size=3, stride=1, padding=1)
        self.output_block = nn.Conv2d(in_channels=self.model_depth, out_channels=self.outChannels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.upgreen0(x)
        x = self.upgreen1(x)
        x = self.upgreen2(x)
        x = self.blue_block(x)
        x = self.output_block(x)
        return x

#AutoEncoder
class AE(BaseModel):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)       
        return decoded

#Classifier
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ModifiedResNet34(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ModifiedResNet34, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 确保输入到全连接层的尺寸正确
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Classifier(BaseModel):
    def __init__(self, **kwargs):
        super(Classifier, self).__init__()
        self.encoder = Encoder(**kwargs)
        self.classify = ModifiedResNet34(BasicBlock, [3, 4, 6, 3], num_classes=2)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        #在通道维度进行拼接
        z = torch.cat([z1, z2], dim = 1)
        return self.classify(z)


#VGG
class VGG(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),           
            
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048*8*8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


#Resnet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):

    def __init__(self, block = Bottleneck, layers = [3, 4, 6, 3], num_classes=2):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


#googlenet
class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1 , kernel_size=1),
            nn.BatchNorm2d(n1x1 ),
            nn.ReLU(True)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3red , kernel_size=1),
            nn.BatchNorm2d(n3x3red ),
            nn.ReLU(True),
            nn.Conv2d(n3x3red , n3x3 , kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True)
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)

class GoogleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(GoogleNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(2, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True)
        )

        self.downsample = nn.MaxPool2d(2, stride=2)

        self.a3 = Inception(96,  32,  48, 64, 8, 16, 16)
        self.b3 = Inception(128, 64, 64, 96, 16, 48, 32)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(240, 96, 48, 104, 8, 24, 32)
        self.b4 = Inception(256, 80, 56, 112, 12, 32, 32)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.downsample(x)
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


#alexnet
class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x