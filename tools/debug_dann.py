import sys
import os
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from models.dann import DANN
from config.config import config

# Dummy feature extractor that returns fixed-size features
class DummyFE(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.feature_dim = feature_dim
    def forward(self, x):
        # ignore input, return deterministic tensor based on batch size
        batch = x.shape[0]
        return torch.rand(batch, self.feature_dim)


def run_debug():
    device = torch.device('cpu')
    batch_size = 4
    num_classes = 10

    fe = DummyFE(feature_dim=config.FEATURE_DIM)
    model = DANN(fe, num_classes).to(device)

    # optimizer including all model parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    criterion_domain = nn.NLLLoss()

    # synthetic inputs (shape doesn't matter for DummyFE)
    src_x = torch.randn(batch_size, 1, 10, 64).to(device)
    tgt_x = torch.randn(batch_size, 1, 10, 64).to(device)

    # domain labels: 0 for source, 1 for target
    src_domains = torch.zeros(batch_size, dtype=torch.long).to(device)
    tgt_domains = torch.ones(batch_size, dtype=torch.long).to(device)

    # 1) List domain_classifier params and requires_grad
    domain_params = [(n, p.requires_grad) for n, p in model.named_parameters() if 'domain_classifier' in n]
    print('domain params (name, requires_grad):')
    for n, r in domain_params:
        print('  ', n, r)

    # 2) Check optimizer contains domain params
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    in_opt = [(n, id(p) in opt_param_ids) for n, p in model.named_parameters() if 'domain_classifier' in n]
    print('domain params in optimizer:')
    for n, included in in_opt:
        print('  ', n, included)

    # 3) Snapshot before
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if 'domain_classifier' in n}

    # Compare forward outputs for alpha=0 and alpha>0
    alpha0 = 0.0
    alpha1 = 0.9
    model.eval()
    with torch.no_grad():
        _, s_dom_a0 = model(src_x, alpha0)
        _, t_dom_a0 = model(tgt_x, alpha0)
        _, s_dom_a1 = model(src_x, alpha1)
        _, t_dom_a1 = model(tgt_x, alpha1)

    print('\nDomain output (first 2 rows) alpha=0 source:')
    print(s_dom_a0[:2])
    print('Domain output (first 2 rows) alpha=0.9 source:')
    print(s_dom_a1[:2])
    print('\nDomain loss values (no training)')
    domain_loss_a0 = criterion_domain(s_dom_a0, src_domains) + criterion_domain(t_dom_a0, tgt_domains)
    domain_loss_a1 = criterion_domain(s_dom_a1, src_domains) + criterion_domain(t_dom_a1, tgt_domains)
    print('  domain_loss alpha=0 :', domain_loss_a0.item())
    print('  domain_loss alpha=0.9 :', domain_loss_a1.item())

    # 4) Backward on domain loss and check grads
    model.train()
    optimizer.zero_grad()
    _, s_dom = model(src_x, alpha1)
    _, t_dom = model(tgt_x, alpha1)
    domain_loss = criterion_domain(s_dom, src_domains) + criterion_domain(t_dom, tgt_domains)
    domain_loss.backward()

    grads = [(n, p.grad is None, (p.grad.detach().cpu().abs().max().item() if p.grad is not None else None))
             for n, p in model.named_parameters() if 'domain_classifier' in n]
    print('\nAfter backward (domain classifier grads):')
    for n, is_none, max_abs in grads:
        print('  ', n, 'grad is None:', is_none, 'max_abs:', max_abs)

    # 5) Step and compare parameters
    optimizer.step()
    after = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if 'domain_classifier' in n}
    changes = {n: (after[n] - before[n]).abs().max().item() for n in before}
    print('\nMax param change per domain param:')
    for n, v in changes.items():
        print('  ', n, v)

    # 6) Label distribution
    print('\nUnique domain labels source:', torch.unique(src_domains), 'target:', torch.unique(tgt_domains))

    print('\nDone debug run')


if __name__ == '__main__':
    run_debug()

