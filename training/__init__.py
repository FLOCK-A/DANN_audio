"""
训练模块
"""
import torch
import torch.nn as nn
import numpy as np
from config.config import config

# 尝试导入 TensorBoard，如果不可用则禁用日志记录
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    class DummyWriter:
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
    SummaryWriter = DummyWriter


def train_dann(model, source_loader, target_loader, val_loader, num_epochs=config.NUM_EPOCHS, 
               device=None, optimizer=None, scheduler=None, optimizer_builder=None, scheduler_builder=None,
               freeze_backbone_epochs=None, freeze_permanent=False, log_dir=config.LOG_DIR, debug=False):
    """
    训练DANN模型
    
    Args:
        model: DANN模型
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        device: 设备 (CPU/GPU)
        optimizer: 可选，已构建的优化器（与 optimizer_builder 二选一）
        scheduler: 可选，已构建的调度器（与 scheduler_builder 二选一）
        optimizer_builder: 可选，可调用对象，接收 model 返回 optimizer
        scheduler_builder: 可调用对象，接收 optimizer 返回 scheduler
        freeze_backbone_epochs: 初始冻结 backbone 的 epoch 数（None 表示使用 config 或不冻结）
        freeze_permanent: 如果 True，永久冻结 backbone（忽略 freeze_backbone_epochs）
        log_dir: 日志保存目录
        debug: 是否启用诊断输出

    Returns:
        dict: 包含训练结果的字典
    """
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 定义损失函数
    criterion_class = nn.NLLLoss()
    criterion_domain = nn.NLLLoss()
    
    # TensorBoard日志记录
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志将保存至: {log_dir}")
    else:
        writer = SummaryWriter()  # 使用虚拟writer
        print("TensorBoard不可用，训练过程将不记录日志")
    
    # 记录最佳准确率
    best_train_acc = 0.0
    best_val_acc = 0.0

    # 处理冻结策略参数
    if freeze_backbone_epochs is None:
        freeze_backbone_epochs = getattr(config, 'FREEZE_BACKBONE_EPOCHS', 0)

    # 如果请求永久冻结（命令行开关），覆盖 freeze_backbone_epochs
    if freeze_permanent:
        freeze_backbone_epochs = float('inf')

    # 如果使用 optimizer_builder，则在这里根据初始冻结状态创建 optimizer/scheduler；否则使用传入的 optimizer/scheduler
    if optimizer_builder is not None:
        # 初始冻结（如果需要）
        if freeze_backbone_epochs and freeze_backbone_epochs > 0:
            # 将 feature_extractor 的参数设为 requires_grad=False
            if hasattr(model, 'feature_extractor'):
                for p in model.feature_extractor.parameters():
                    p.requires_grad = False
                print(f"Backbone frozen for first {freeze_backbone_epochs} epochs")
        optimizer = optimizer_builder(model)
        scheduler = scheduler_builder(optimizer) if scheduler_builder is not None else None
    else:
        if optimizer is None:
            raise ValueError("optimizer must be provided to train_dann if optimizer_builder is not supplied")

    # 训练循环
    for epoch in range(num_epochs):
        # 在 epoch 开始时检查是否达到解冻时刻（注意：epoch 从 0 开始）
        if optimizer_builder is not None and (not freeze_permanent):
            # 当 epoch == freeze_backbone_epochs 时解冻（例如 freeze_backbone_epochs=5 则第 6 轮开始训练 backbone）
            if freeze_backbone_epochs != float('inf') and epoch == int(freeze_backbone_epochs):
                if hasattr(model, 'feature_extractor'):
                    for p in model.feature_extractor.parameters():
                        p.requires_grad = True
                    print(f"Unfroze backbone at epoch {epoch}. Rebuilding optimizer and scheduler to include backbone with lower LR.")
                    # 重新构建 optimizer 和 scheduler，使 backbone 参数被包含并使用正确的 lr
                    optimizer = optimizer_builder(model)
                    scheduler = scheduler_builder(optimizer) if scheduler_builder is not None else None

        model.train()
        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        correct_class = 0
        total_samples = 0
        
        # 获取数据迭代器
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # 计算每个epoch的批次数量
        num_batches = min(len(source_loader), len(target_loader))
        
        # One-time debug flag to run diagnostics on the very first batch when requested
        debug_done = False
        for i in range(num_batches):
            # 获取源域和目标域数据
            try:
                source_data = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_data = next(source_iter)
                
            try:
                target_data = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data = next(target_iter)
            
            # 获取数据
            source_features = source_data['features'].to(device)
            source_labels = source_data['label'].to(device)
            source_domains = source_data['domain'].to(device)
            
            target_features = target_data['features'].to(device)
            target_domains = target_data['domain'].to(device)
            
            # 计算进度 p 和 alpha（修正除法顺序，p in [0,1]）
            total_iters = max(1, num_epochs * num_batches)
            cur_iter = epoch * num_batches + i
            p = float(cur_iter) / float(total_iters)
            alpha = (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

            # 清零梯度
            if optimizer is None:
                raise ValueError("optimizer must be provided to train_dann")
            optimizer.zero_grad()
            
            # 源域前向传播
            source_class_output, source_domain_output = model(source_features, alpha)
            
            # 目标域前向传播
            _, target_domain_output = model(target_features, alpha)
            
            # 计算损失
            class_loss = criterion_class(source_class_output, source_labels)
            # 推荐一次性拼接域预测，避免批大小/权重问题
            domain_preds = torch.cat([source_domain_output, target_domain_output], dim=0)
            domain_labels = torch.cat([source_domains, target_domains], dim=0).to(domain_preds.device).long()
            domain_loss = criterion_domain(domain_preds, domain_labels)

            # 使用配置中的权重来控制域损失在总损失中的影响
            loss = class_loss + config.LAMBDA_DOMAIN * domain_loss

            # 如果开启 debug，执行一组诊断检查（仅在训练开始的第一个 batch 执行一次）
            if debug and (not debug_done):
                print('\n=== DANN debug diagnostics (running once) ===')

                # 1. 列出 domain_classifier 的参数名与 requires_grad
                domain_params = [(n, p.requires_grad) for n, p in model.named_parameters() if 'domain_classifier' in n]
                print('domain params (name, requires_grad):', domain_params)

                # 2. 检查 optimizer 是否包含 domain_classifier 的参数
                opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
                in_opt = [(n, id(p) in opt_param_ids) for n, p in model.named_parameters() if 'domain_classifier' in n]
                print('domain params in optimizer:', in_opt)

                # --- 新增：打印 optimizer param_groups 的信息，并断言覆盖所有可训练参数 ---
                try:
                    print('\nOptimizer param_groups info:')
                    total_params_in_groups = 0
                    for idx, g in enumerate(optimizer.param_groups):
                        group_lr = g.get('lr', None)
                        group_wd = g.get('weight_decay', None)
                        group_param_count = len(g.get('params', []))
                        total_params_in_groups += group_param_count
                        print(f"  group {idx}: lr={group_lr}, weight_decay={group_wd}, params={group_param_count}")

                    total_trainable = len([p for p in model.parameters() if p.requires_grad])
                    print(f"Total trainable params: {total_trainable}, Total params in optimizer: {total_params_in_groups}")
                    if total_params_in_groups != total_trainable:
                        print("Warning: optimizer param_groups do not cover all trainable parameters.")
                        # 列出未覆盖的参数名（通过 id 匹配）
                        covered_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
                        missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in covered_ids]
                        print('Missing params:', missing)
                except Exception as e:
                    print('Could not introspect optimizer param_groups:', e)

                # 3. 记录 domain_classifier 参数的一个快照（step 之前）
                before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if 'domain_classifier' in n}

                # 前向、域损失、反向
                optimizer.zero_grad()
                _, s_dom = model(source_features, alpha)
                _, t_dom = model(target_features, alpha)
                domain_loss_debug = criterion_domain(s_dom, source_domains) + criterion_domain(t_dom, target_domains)
                domain_loss_debug.backward()

                # 4. 检查 domain_classifier 参数的梯度（是否为 None 或全 0）
                grads = [(n, p.grad is None, (p.grad.detach().cpu().abs().max().item() if p.grad is not None else None))
                         for n, p in model.named_parameters() if 'domain_classifier' in n]
                print('domain grads (is None, max_abs):', grads)

                # 5. 执行一步优化并比对参数是否改变
                optimizer.step()
                after = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if 'domain_classifier' in n}
                changes = {n: (after[n] - before[n]).abs().max().item() for n in before}
                print('max param change per domain param:', changes)

                # 6. 额外检查 domain 标签分布
                try:
                    src_unique = torch.unique(source_domains)
                    tgt_unique = torch.unique(target_domains)
                except Exception:
                    src_unique = source_domains
                    tgt_unique = target_domains
                print('unique domain labels source:', src_unique, 'target:', tgt_unique)
                print('=== End diagnostics ===\n')

                # 标记已做诊断，避免重复
                debug_done = True

                # 继续下一个 batch (diagnostics already performed using backward+step)
                continue

            # 反向传播和优化（如果 debug 已经对本 batch 做过 backward/step，则仍然继续正常训练；
            # 这会在 debug=True 情况下导致该第一个 batch 执行两次 step — acceptable for quick diagnostics.）
            loss.backward()
            optimizer.step()

            # 统计信息（累加用于 epoch 级别的汇报）
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()

            pred = source_class_output.argmax(dim=1, keepdim=True)
            correct_class += pred.eq(source_labels.view_as(pred)).sum().item()
            total_samples += source_features.size(0)

        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            # scheduler.get_last_lr() 在不同 scheduler 中可能返回多个值，取第一个作为展示
            try:
                current_lr = scheduler.get_last_lr()[0]
            except Exception:
                current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # 计算平均损失和准确率
        avg_loss = total_loss / num_batches
        avg_class_loss = total_class_loss / num_batches
        avg_domain_loss = total_domain_loss / num_batches
        class_accuracy = 100. * correct_class / total_samples
        
        # 验证阶段（使用 clip-level 聚合）
        val_accuracy_clip = evaluate_model_clip_level(model, val_loader, device)

        # 更新最佳准确率（以 clip-level val 作为参考）
        if class_accuracy > best_train_acc:
            best_train_acc = class_accuracy
        if val_accuracy_clip > best_val_acc:
            best_val_acc = val_accuracy_clip

        # 记录TensorBoard日志（包含 clip-level 验证精度）
        writer.add_scalar('Training Loss', avg_loss, epoch)
        writer.add_scalar('Classification Loss', avg_class_loss, epoch)
        writer.add_scalar('Domain Loss', avg_domain_loss, epoch)
        writer.add_scalar('Training Accuracy (sample)', class_accuracy, epoch)
        writer.add_scalar('Validation Accuracy (clip)', val_accuracy_clip, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # 打印统计信息
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {avg_loss:.4f}, '
              f'Class Loss: {avg_class_loss:.4f}, '
              f'Domain Loss: {avg_domain_loss:.4f}, '
              f'Train Acc: {class_accuracy:.2f}%, '
              f'Val Acc (clip): {val_accuracy_clip:.2f}%, '
              f'LR: {current_lr:.6f}')

    writer.close()
    
    # 返回训练结果
    return {
        'best_train_acc': best_train_acc,
        'best_val_acc': best_val_acc
    }


def evaluate_model(model, data_loader, device):
    """
    评估模型性能
    
    Args:
        model: 待评估的模型
        data_loader: 数据加载器
        device: 设备 (CPU/GPU)
        
    Returns:
        accuracy: 准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in data_loader:
            features = data['features'].to(device)
            labels = data['label'].to(device)
            
            # 前向传播 (alpha=0表示不进行域对抗)
            class_output, _ = model(features, alpha=0)
            
            # 计算准确率
            pred = class_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += features.size(0)
    
    accuracy = 100. * correct / total
    model.train()
    return accuracy


def test_model(model, test_loader, device):
    """
    在测试集上评估模型
    
    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 设备 (CPU/GPU)
        
    Returns:
        accuracy: 准确率
    """
    # 默认执行 clip-level 聚合评估（mean log-prob pooling）
    return evaluate_model_clip_level(model, test_loader, device)


def evaluate_model_clip_level(model, data_loader, device):
    """
    Clip-level evaluation by aggregating 1s-segment log-probs into 10s clip predictions.

    Aggregation strategy:
      - For each sample the dataset must expose 'clip_id' and the model outputs log-prob (log-softmax)
      - Collect per-segment log-probs per clip_id, compute mean(logp) across segments, then argmax

    Returns clip-level accuracy (percentage)
    """
    model.eval()
    clip_logits = {}  # clip_id -> list of logp tensors (C,)
    clip_labels = {}  # clip_id -> ground truth label (int)

    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            # Expect dataset to provide clip_id per sample; after batching it's a list
            clip_ids = batch.get('clip_id', None)
            if clip_ids is None:
                # If clip_id missing, fall back to file names
                clip_ids = batch.get('file', None)
            # Forward pass (alpha=0 to disable domain adversarial)
            class_output, _ = model(features, alpha=0)
            # class_output is log-prob (N, C)
            # Iterate over batch items
            for i, cid in enumerate(clip_ids):
                # If DataLoader collates strings into a tuple/list, ensure we have a str
                if isinstance(cid, (list, tuple)):
                    cur_cid = cid[0]
                else:
                    cur_cid = cid
                logp = class_output[i].detach().cpu()
                lab = int(labels[i].detach().cpu().item())
                if cur_cid not in clip_logits:
                    clip_logits[cur_cid] = [logp]
                    clip_labels[cur_cid] = lab
                else:
                    clip_logits[cur_cid].append(logp)

    # Aggregate per clip
    correct = 0
    total = 0
    for cid, logs in clip_logits.items():
        # Stack and mean in log-domain
        stacked = torch.stack(logs, dim=0)  # (T, C)
        mean_logp = stacked.mean(dim=0)  # (C,)
        pred = mean_logp.argmax().item()
        true = clip_labels.get(cid, None)
        if true is None:
            continue
        if pred == true:
            correct += 1
        total += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    model.train()
    return accuracy
