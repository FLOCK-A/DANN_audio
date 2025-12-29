import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
            
        return x


class ProjectionHead(nn.Module):
    """投影头，用于特征投影"""
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=2048):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_layer(m)
            elif isinstance(m, nn.BatchNorm1d):
                init_bn(m)
    
    def forward(self, x):
        x = self.head(x)
        return x


class Cnn14LogMel(nn.Module):
    def __init__(self, classes_num=10, checkpoint_path=None, use_projection_head=False):
        super().__init__()
        self.use_projection_head = use_projection_head
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.bn1 = nn.BatchNorm1d(2048)
        
        # 添加投影头
        if self.use_projection_head:
            self.projection_head = ProjectionHead(input_dim=2048, hidden_dim=2048, output_dim=2048)

        self.init_weight()
        
        # 如果提供了检查点路径，则加载预训练权重
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                # 处理不同的检查点格式
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 过滤掉不匹配的键（因为我们的模型结构略有不同）
                filtered_state_dict = {}
                for k, v in state_dict.items():
                    # 移除可能存在的module.前缀
                    k = k.replace('module.', '')
                    # 只保留与当前模型匹配的键
                    if k in self.state_dict():
                        filtered_state_dict[k] = v
                
                self.load_state_dict(filtered_state_dict, strict=False)
                print(f"成功加载预训练权重: {checkpoint_path}")
            except Exception as e:
                print(f"加载预训练权重失败: {e}")

    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn1)
        
        # 初始化投影头权重
        if self.use_projection_head:
            self.projection_head.init_weights()
 
    def forward(self, logmel_input):
        """
        Input: (batch, 1, time_steps, n_mels)
        Output: (batch, 2048)
        """
        # 根据输入维度进行处理
        if logmel_input.dim() == 3:
            # 如果是3D输入 (batch, time_steps, n_mels)，添加通道维度
            logmel_input = logmel_input.unsqueeze(1)
        elif logmel_input.dim() == 4:
            # 如果已经是4D输入 (batch, channels, time_steps, n_mels)
            pass
        else:
            raise ValueError(f"意外的输入维度: {logmel_input.dim()}, 形状: {logmel_input.shape}")
        
        # logmel_input: (batch, 1, time_steps, n_mels)
        # 调整维度以匹配模型期望的输入
        x = logmel_input.transpose(2, 3)  # (batch, 1, n_mels, time_steps)
        
        # 通过卷积块处理
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        # 全局池化
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 应用投影头（如果启用）
        if self.use_projection_head:
            x = self.projection_head(x)
        
        return x

    def get_backbone_state_dict(self):
        """
        获取backbone的state_dict（不包括投影头）
        """
        backbone_state_dict = {}
        for k, v in self.state_dict().items():
            if not k.startswith('projection_head'):
                backbone_state_dict[k] = v
        return backbone_state_dict

    def get_projection_head_state_dict(self):
        """
        获取投影头的state_dict
        """
        if not self.use_projection_head:
            return None
            
        projection_head_state_dict = {}
        for k, v in self.state_dict().items():
            if k.startswith('projection_head'):
                projection_head_state_dict[k] = v
        return projection_head_state_dict