import torch
from torch import nn

act = nn.ReLU(inplace=True)

def conv33(input, output, stride=1, groups=1, padding=1):
    """
    This is a 3x3 convolutional layer with groups = 1 and padding = 1 as default
    This will be widely used in resnet
    """
    conv = nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=padding, groups=groups,bias=False, dilation=padding)
    return conv

def conv11(input, output, stride=1):
    """
    in resnet, conv1x1 is widely used to modify dimension
    """
    conv = nn.Conv2d(input, output, kernel_size=1, stride=stride,bias=False)
    return conv

class Block_resnet(nn.Module):
    enlarge = 1
    def __init__(self, input, output, stride=1, padding=1, down=None, normalization=None, activate=act, groups=1):
        super(Block_resnet, self).__init__()
        if not normalization:
            normalization = nn.BatchNorm2d
        self.conv_1 = conv33(input=input, output=output, stride=stride)
        self.batch_norm_1 = normalization(output)
        self.activate = activate
        self.conv_2 = conv33(input=output, output=output)
        self.batch_norm_2 = normalization(output)
        self.down = down
        self.stride = stride

    def forward(self, x):
        history = x
        # we will add the residual in forward function
        res = self.conv_1(x)
        res = self.batch_norm_1(res)
        res = self.activate(res)
        res = self.conv_2(res)
        res = self.batch_norm_2(res)
        if self.down:
            # ensure the dimension of history and output
            # so that we can add the residual
            # if the dimension of result has changed, we shall do downsampling on history
            history = self.down(history)
        res += history
        return self.activate(res)

class ResNet(nn.Module):
    '''
    Original resnet without any improvement
    '''

    def __init__(self, Block, layer_list, n_class=1000, groups=1, init_as_zero=False, normalization=None, activate=act):
        super(ResNet, self).__init__()
        if not normalization:
            normalization = nn.BatchNorm2d
        self.normalization = normalization
        self.input_channel = 64
        self.padding = 1
        self.groups = groups

        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm_1 = self.normalization(self.input_channel)
        self.activate = activate

        channels = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        self.rest_layers = []
        for ind, layer_num in enumerate(layer_list):
            channel = channels[ind]
            stride = strides[ind]
            down = None
            pre_padding = self.padding
            if stride!=1 or self.input_channel!=channel*Block.enlarge:
                down = nn.Sequential(
                    conv11(self.input_channel, channel*Block.enlarge, stride),
                    self.normalization(channel*Block.enlarge),
                )
            append_layers = []
            append_layers.append(Block(self.input_channel, channel, stride=stride, down=down,
                                       groups=self.groups, padding=pre_padding, normalization=self.normalization))
            self.input_channel = channel * Block.enlarge
            for i in range(1, layer_num):
                append_layers.append(Block(self.input_channel, channel, 
                                           groups=self.groups, padding=self.padding,
                                           normalization=self.normalization))
            self.rest_layers.append(nn.Sequential(*append_layers))
        self.layer1 = self.rest_layers[0]
        self.layer2 = self.rest_layers[1]
        self.layer3 = self.rest_layers[2]
        self.layer4 = self.rest_layers[3]
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        l = channels[len(append_layers)-1]
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(l*Block.enlarge, n_class)

        # Simple initialization, this part was inspired by some open source code
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if init_as_zero:
            for module in self.modules():
                if isinstance(module, Block_resnet):
                    nn.init.constant_(module.batch_norm_2.weight, 0)



    def forward(self, x):
        res = self.conv_1(x)
        res = self.batch_norm_1(res)
        res = self.activate(res)
        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.layer4(res)
        res = self.pooling(res)
        res = torch.flatten(res, 1)
        res = self.dropout(res)
        res = self.fc(res)
        return res

layer_number_record = {
    "14":[2,2,2],
    "18": [2,2,2,2],
    "34": [3,4,6,3]
}

def ResNet14():
    model = ResNet(Block_resnet, layer_number_record['14'])
    return model

def ResNet18():
    model = ResNet(Block_resnet, layer_number_record['18'])
    return model

def ResNet34():
    model = ResNet(Block_resnet, layer_number_record['34'])
    return model







