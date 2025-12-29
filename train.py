"""
主训练脚本
"""
import os
import torch
import argparse
import json
import webbrowser
import subprocess
import time
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from models.dann import DANN
from models.feature_extractor import Cnn14LogMel as Cnn14
from data.dataloader import get_dataloader
from training import train_dann, test_model
from config.config import config


def get_optimizer(optimizer_name, model, lr_head, lr_backbone, weight_decay=0.0, exclude_from_wd=False):
    """获取优化器，按 backbone / head 分组并支持 AdamW

    Args:
        optimizer_name: 'adam' | 'sgd' | 'adamw'
        model: nn.Module (期望拥有 feature_extractor, class_classifier, domain_classifier)
        lr_head: float, head (class + domain) 学习率
        lr_backbone: float, backbone 学习率
        weight_decay: float, weight decay（AdamW 使用）
        exclude_from_wd: bool, 是否对 bias/LayerNorm/BatchNorm 等参数禁用 weight decay
    """
    # 收集参数
    backbone_params = []
    head_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('feature_extractor'):
            backbone_params.append(p)
        elif name.startswith('class_classifier') or name.startswith('domain_classifier'):
            head_params.append(p)
        else:
            other_params.append((name, p))

    # 将未分类的参数放到 head（更安全，避免漏掉训练参数）
    if other_params:
        other_params_only = [p for n, p in other_params]
        head_params.extend(other_params_only)

    param_groups = []

    # 可选地排除 bias 和 BatchNorm/LayerNorm 的 weight decay
    if exclude_from_wd and weight_decay > 0.0:
        def _partition_no_decay(params, prefix):
            decay, no_decay = [], []
            for n, p in params:
                if len(p.shape) == 1 or n.endswith('.bias'):
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        # 需要原始名字-参数对来判定，因此重新收集带名字的列表
        backbone_named = [(n, p) for n, p in model.named_parameters() if n.startswith('feature_extractor') and p.requires_grad]
        head_named = [(n, p) for n, p in model.named_parameters() if (n.startswith('class_classifier') or n.startswith('domain_classifier')) and p.requires_grad]
        # add other_named
        other_named = [(n, p) for n, p in model.named_parameters() if (not n.startswith('feature_extractor') and not n.startswith('class_classifier') and not n.startswith('domain_classifier')) and p.requires_grad]
        head_named.extend(other_named)

        b_decay, b_no_decay = _partition_no_decay(backbone_named, 'backbone')
        h_decay, h_no_decay = _partition_no_decay(head_named, 'head')

        if b_decay:
            param_groups.append({'params': b_decay, 'lr': lr_backbone, 'weight_decay': weight_decay})
        if b_no_decay:
            param_groups.append({'params': b_no_decay, 'lr': lr_backbone, 'weight_decay': 0.0})
        if h_decay:
            param_groups.append({'params': h_decay, 'lr': lr_head, 'weight_decay': weight_decay})
        if h_no_decay:
            param_groups.append({'params': h_no_decay, 'lr': lr_head, 'weight_decay': 0.0})
    else:
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay})
        if head_params:
            param_groups.append({'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay})

    # 选择优化器
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adamw':
        return AdamW(param_groups)
    elif optimizer_name == 'adam':
        # Adam 也接受 weight_decay 参数，但一般建议用 AdamW
        # 注意：torch.optim.Adam 接受 weight_decay kwarg
        return Adam(param_groups)
    elif optimizer_name == 'sgd':
        # 对于 SGD，我们传递 momentum=0.9
        return SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(scheduler_name, optimizer, num_epochs, warmup_epochs=0):
    """获取学习率调度器"""
    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import LambdaLR
        def warmup_lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性warmup
                return float(epoch + 1) / float(warmup_epochs + 1)
            else:
                # 之后正常调度
                if scheduler_name == 'cosine':
                    import math
                    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
                elif scheduler_name == 'step':
                    # 简化的step衰减
                    return 0.1 ** ((epoch - warmup_epochs) // (num_epochs // 3))
                elif scheduler_name == 'exp':
                    return 0.9 ** (epoch - warmup_epochs)
                else:
                    return 1.0
        return LambdaLR(optimizer, warmup_lr_lambda)
    # 不使用warmup
    if scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'step':
        return StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)
    elif scheduler_name == 'exp':
        return ExponentialLR(optimizer, gamma=0.9)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def save_model(model, training_mode, use_projection_head=False):
    """保存模型"""
    print('Saving models ...')
    save_folder = 'trained_models'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 分别保存特征提取器backbone和投影头、标签分类器和域判别器的权重
    if use_projection_head:
        # 分别保存backbone和投影头
        backbone_state_dict = model.feature_extractor.get_backbone_state_dict()
        projection_head_state_dict = model.feature_extractor.get_projection_head_state_dict()
        
        torch.save(backbone_state_dict, f'trained_models/backbone_{training_mode}.pt')
        torch.save(projection_head_state_dict, f'trained_models/projection_head_{training_mode}.pt')
        print(f"Backbone and projection head saved separately.")
    else:
        # 保存整个特征提取器
        torch.save(model.feature_extractor.state_dict(), f'trained_models/feature_extractor_{training_mode}.pt')
    
    torch.save(model.class_classifier.state_dict(), f'trained_models/class_classifier_{training_mode}.pt')
    torch.save(model.domain_classifier.state_dict(), f'trained_models/domain_classifier_{training_mode}.pt')
    
    print('The model has been successfully saved!')


def start_tensorboard(log_dir):
    """启动TensorBoard"""
    try:
        # 启动TensorBoard作为后台进程
        tensorboard_process = subprocess.Popen([
            'tensorboard', '--logdir', log_dir, '--host', 'localhost', '--port', '6006'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待一点时间让TensorBoard启动
        time.sleep(3)
        
        # 在浏览器中打开TensorBoard
        webbrowser.open('http://localhost:6006')
        
        print("TensorBoard已在浏览器中打开: http://localhost:6006")
        return tensorboard_process
    except Exception as e:
        print(f"无法启动TensorBoard: {e}")
        return None


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train DANN for ASC')
    parser.add_argument('--dataset_json', type=str, default=r'E:\代码\data2\dann_dataset_b.json',help='数据集JSON文件，包含train/val/test三部分')
    parser.add_argument('--data_root', type=str, default=r'E:\代码\data2\raw', help='数据根目录')
    parser.add_argument('--checkpoint_path', type=str, default=r'E:\代码\DANN_audio\Cnn14_16k_mAP=0.438.pth',help='预训练模型检查点路径')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=config.NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='学习率 (默认用于 head)')
    parser.add_argument('--lr_backbone', type=float, default=config.LR_BACKBONE, help='backbone 的学习率（默认 1e-4，推荐 3e-5~5e-5）')
    parser.add_argument('--lr_head', type=float, default=None, help='head 的学习率（默认使用 --lr 或 config.LR_HEAD）')
    parser.add_argument('--optimizer', type=str, default=config.OPTIMIZER_DEFAULT,
                        choices=['adam', 'sgd', 'adamw'], help='优化器类型')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'exp'], help='学习率调度器类型')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup轮数')
    parser.add_argument('--save_model', type=str, default=config.MODEL_SAVE_PATH, help='模型保存路径')
    parser.add_argument('--freeze_feature_extractor', action='store_true', 
                        help='是否冻结特征提取器参数')
    parser.add_argument('--use_projection_head', action='store_true',
                        help='是否在特征提取器中使用投影头')
    parser.add_argument('--open_tensorboard', action='store_true',
                        help='是否在训练开始时自动打开TensorBoard')
    parser.add_argument('--debug', action='store_true', help='启用训练诊断输出（仅在第一个 batch 运行诊断）')
    args = parser.parse_args()

    # 更新配置
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.num_epochs
    config.LEARNING_RATE = args.lr
    if args.save_model:
        config.MODEL_SAVE_PATH = args.save_model

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 启动TensorBoard（如果需要）
    tensorboard_process = None
    if args.open_tensorboard:
        tensorboard_process = start_tensorboard(config.LOG_DIR)

    # 加载数据集JSON文件
    with open(args.dataset_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 创建特征提取器 (使用官方预训练的CNN14)
    feature_extractor = Cnn14(classes_num=args.num_classes, 
                             checkpoint_path=args.checkpoint_path,
                             use_projection_head=args.use_projection_head)
    

    # 创建DANN模型
    model = DANN(feature_extractor, args.num_classes, dropout=config.DROPOUT)

    # 构建 optimizer/scheduler 的 builder（延迟创建，以便在解冻时重新构建）
    lr_head = args.lr_head if args.lr_head is not None else args.lr

    def optimizer_builder(m):
        # get_optimizer 会基于 m.named_parameters() 中的 requires_grad 自动包含/排除 backbone
        return get_optimizer(args.optimizer, m, lr_head=lr_head, lr_backbone=args.lr_backbone, weight_decay=config.WEIGHT_DECAY, exclude_from_wd=True)

    def scheduler_builder(opt):
        return get_scheduler(args.scheduler, opt, args.num_epochs, args.warmup_epochs)

    # 加载数据
    # train是源域数据，val是目标域数据，test是测试集
    source_loader = get_dataloader(
        samples=dataset['train'],
        data_root=args.data_root,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    target_loader = get_dataloader(
        samples=dataset['val'],
        data_root=args.data_root,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = get_dataloader(
        samples=dataset['val'],
        data_root=args.data_root,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    test_loader = get_dataloader(
        samples=dataset['test'],
        data_root=args.data_root,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # 训练模型（传入 builder，让 training 模块在解冻时重新构建 optimizer/scheduler）
    train_result = train_dann(
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        device=device,
        optimizer_builder=optimizer_builder,
        scheduler_builder=scheduler_builder,
        freeze_backbone_epochs=None,  # None -> training 模块将使用 config.FREEZE_BACKBONE_EPOCHS
        freeze_permanent=args.freeze_feature_extractor,
        debug=args.debug
    )

    # 测试模型
    test_accuracy = test_model(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # 保存模型
    save_model(model, "dann_asc", args.use_projection_head)
    
    # 记录最终的准确率
    print(f"Final Results:")
    print(f"  Best Training Accuracy: {train_result.get('best_train_acc', 0):.2f}%")
    print(f"  Best Validation Accuracy: {train_result.get('best_val_acc', 0):.2f}%")
    print(f"  Test Accuracy: {test_accuracy:.2f}%")

    # 关闭TensorBoard进程（如果启动了）
    if tensorboard_process:
        tensorboard_process.terminate()


if __name__ == '__main__':
    main()