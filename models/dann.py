"""
DANN模型实现
"""
import torch
import torch.nn as nn
from torch.autograd import Function
from config.config import config


class ReverseLayer(Function):
    """
    梯度反转层
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    """
    Domain Adversarial Neural Network for Acoustic Scene Classification
    """

    def __init__(self, feature_extractor, num_classes, dropout=config.DROPOUT):
        """
        初始化DANN模型
        
        Args:
            feature_extractor: 特征提取器 (预训练的CNN14)
            num_classes: 分类数量
            dropout: Dropout比例
        """
        super(DANN, self).__init__()
        self.feature_extractor = feature_extractor
        self.dropout = nn.Dropout(dropout)
        
        # 获取特征维度 (假设为2048，需根据实际CNN14输出调整)
        feature_dim = config.FEATURE_DIM
        
        # 标签分类器
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        # 域判别器
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha=0.0):
        """
        前向传播
        
        Args:
            input_data: 输入数据
            alpha: 梯度反转的系数（仅用于 GRL，影响反向回到 feature_extractor 的梯度）

        Returns:
            output_label: 标签预测结果
            output_domain: 域预测结果
        """
        # 提取特征
        feature = self.feature_extractor(input_data)
        feature = feature.view(feature.size(0), -1)
        feature = self.dropout(feature)
        
        # 反向梯度层
        reverse_feature = ReverseLayer.apply(feature, alpha)
        
        # 分类器输出
        output_label = self.class_classifier(feature)
        
        # 域判别器输出
        output_domain = self.domain_classifier(reverse_feature)
        
        return output_label, output_domain