"""
配置文件，包含模型和训练的所有超参数
"""

class Config:
    # 数据相关参数
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    # 模型相关参数
    DROPOUT = 0.3
    FEATURE_DIM = 2048  # CNN14特征维度
    
    # 训练相关参数
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3 #默认学习率，已废弃
    WEIGHT_DECAY = 1e-4
    # 新增：分组学习率（backbone / head）
    LR_BACKBONE = 3e-5  # 推荐范围: 3e-5 ~ 5e-5（更保守）
    LR_HEAD = 1e-3      # head 的默认学习率（可由 --lr 或 --lr_head 覆盖）
    OPTIMIZER_DEFAULT = 'adamw'  # 推荐使用 AdamW

    # 新增：冻结 backbone 的初始 epoch 数（0 表示不冻结）
    FREEZE_BACKBONE_EPOCHS = 5  # 推荐 5-10

    # 域对抗参数
    LAMBDA_DOMAIN = 1.0  # 域对抗损失的权重 (建议 >=0, kept separate from GRL alpha)

    # 其他参数
    LOG_DIR = 'E:\代码\DANN_audio\output'
    MODEL_SAVE_PATH = 'E:\代码\DANN_audio\output'


# 为方便访问，创建一个全局配置实例
config = Config()