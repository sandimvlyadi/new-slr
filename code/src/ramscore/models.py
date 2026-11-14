"""Backbones and classification/ordinal heads for RAM-H1200 & RheumaVIT experiments (RQ0-RQ6).

Supported heads:
  - 'softmax':          K logits (CE, Balanced Softmax, Focal, LDAM-DRW, Soft Labels, GCE)
  - 'ordinal':          K-1 cumulative logits P(y > k) (Coral / Frank-Hall / FOL)
  - 'corn':             K-1 conditional logits P(y > k | y >= k) (Shi et al., 2023)
  - 'hierarchical':     1 binary logit P(y >= 1) + (K-1) conditional severity logits
  - 'cumulative_link':  Proportional Odds / Cumulative-Link Model (M089): 1 latent score + K-1 monotone cutpoints
  - 'softmax_corn':     K softmax logits + (K-1) CORN auxiliary logits (RQ2-16)
"""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_OUT_DIM = {
    'softmax': lambda k: k,
    'ordinal': lambda k: k - 1,
    'corn': lambda k: k - 1,
    'hierarchical': lambda k: k,
    'cumulative_link': lambda k: 1,
    'softmax_corn': lambda k: 2 * k - 1,
}


class CumulativeLinkHead(nn.Module):
    """Proportional Odds / Cumulative-Link Ordinal Head (McCullagh 1980; M089).

    Projects features h(x) to a scalar latent severity score eta(x) = w^T h(x),
    and learns K-1 strictly increasing cutpoints theta_0 < theta_1 < ... < theta_{K-2}.
    Forward returns log-probabilities log P(Y = c | x) of shape (B, K).
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(in_features, 1, bias=False)
        self.theta0 = nn.Parameter(torch.tensor(-1.0))
        self. log_deltas = nn.Parameter(torch.zeros(num_classes - 2))

    def cutpoints(self) -> torch.Tensor:
        deltas = F.softplus(self.log_deltas) + 1e-4
        return torch.cat([self.theta0[None], self.theta0 + torch.cumsum(deltas, dim=0)])

    def probs(self, eta: torch.Tensor) -> torch.Tensor:
        theta = self.cutpoints()  # (K-1,)
        cdf = torch.sigmoid(theta[None, :] - eta)  # (B, K-1): P(Y <= k)
        zeros = torch.zeros_like(eta)
        ones = torch.ones_like(eta)
        cdf_ext = torch.cat([zeros, cdf, ones], dim=1)  # (B, K+1)
        p = (cdf_ext[:, 1:] - cdf_ext[:, :-1]).clamp_min(1e-7)
        return p / p.sum(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta = self.proj(self.drop(x))
        return torch.log(self.probs(eta))


class CumulativeLinkWrapper(nn.Module):
    """Wraps a timm backbone (num_classes=0) with CumulativeLinkHead."""

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.head = CumulativeLinkHead(backbone.num_features, num_classes, dropout=dropout)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward_head(self, feats: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        pooled = self.backbone.forward_head(feats, pre_logits=True)
        return pooled if pre_logits else self.head(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone(x)
        return self.head(pooled)


def build_model(
    name: str = 'resnet34',
    num_classes: int = 5,
    head: str = 'ordinal',
    pretrained: bool = True,
    in_chans: int = 1,
    drop_rate: float = 0.0,
) -> nn.Module:
    if head not in HEAD_OUT_DIM:
        raise ValueError(f'Unknown head {head!r}; choose from {list(HEAD_OUT_DIM)}')
    if head == 'cumulative_link':
        backbone = timm.create_model(name, pretrained=pretrained, in_chans=in_chans, num_classes=0)
        return CumulativeLinkWrapper(backbone, num_classes=num_classes, dropout=drop_rate)
    out_dim = HEAD_OUT_DIM[head](num_classes)
    return timm.create_model(name, pretrained=pretrained, in_chans=in_chans, num_classes=out_dim, drop_rate=drop_rate)


def load_backbone(model: nn.Module, path: str) -> list[str]:
    """Load encoder weights from a pretrain.py checkpoint into a build_model() network."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = state.get('model', state)
    target = model.backbone if isinstance(model, CumulativeLinkWrapper) else model
    missing, unexpected = target.load_state_dict(state, strict=False)
    head_prefix = getattr(target, 'pretrained_cfg', {}).get('classifier', 'head') + '.'
    bad_missing = [k for k in missing if not k.startswith(head_prefix) and not k.startswith('head.')]
    bad_unexpected = [k for k in unexpected if not k.startswith(head_prefix) and not k.startswith('head.')]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f'Backbone mismatch loading {path}: missing={bad_missing}, unexpected={bad_unexpected}')
    return missing


def load_full(model: nn.Module, path: str) -> None:
    """Load backbone and classification head from a train.py checkpoint (stage 2 of LP-FT)."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('model', state), strict=True)


# ---------------------------------------------------------------------- Losses

def _reduce_weighted(loss_vec: torch.Tensor, sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    if sample_weights is None:
        return loss_vec.mean()
    w = sample_weights.to(loss_vec.device, dtype=loss_vec.dtype)
    return (loss_vec * w).sum() / w.sum().clamp_min(1e-8)


def ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=labels.device)[None, :]
    return (labels[:, None] > thresholds).float()


def ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    sample_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Binary cross-entropy on cumulative thresholds P(y > k), with optional Focal Ordinal modulation (RQ2-11, M132)."""
    t = ordinal_targets(labels, num_classes)
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction='none')  # (B, K-1)
    if focal_gamma > 0.0:
        p = torch.sigmoid(logits)
        pt = torch.where(t > 0.5, p, 1.0 - p)
        thresholds = torch.arange(num_classes - 1, device=labels.device, dtype=logits.dtype)[None, :]
        dist_w = 1.0 + (labels[:, None].float() - thresholds).abs() / max(num_classes - 1, 1)
        bce = ((1.0 - pt) ** focal_gamma) * dist_w * bce
    per_sample = bce.mean(dim=1)
    return _reduce_weighted(per_sample, sample_weights)


def corn_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    tau: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """CORN loss (Shi et al., 2023) with optional conditional logit adjustment tau_k."""
    if tau is not None:
        logits = logits + tau[None, :]
    total = logits.new_tensor(0.0)
    n_active = 0
    for k in range(num_classes - 1):
        mask = labels >= k
        if not mask.any():
            continue
        target = (labels[mask] > k).float()
        bce = F.binary_cross_entropy_with_logits(logits[mask, k], target, reduction='none')
        w = sample_weights[mask] if sample_weights is not None else None
        total = total + _reduce_weighted(bce, w)
        n_active += 1
    return total / max(n_active, 1)


def hierarchical_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    sev_weight: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    is_pos = (labels >= 1).float()
    pw = pos_weight.to(logits.device) if pos_weight is not None else None
    l_bin_vec = F.binary_cross_entropy_with_logits(logits[:, 0], is_pos, pos_weight=pw, reduction='none')
    l_bin = _reduce_weighted(l_bin_vec, sample_weights)
    pos_mask = labels >= 1
    if not pos_mask.any():
        return l_bin
    sw = sev_weight.to(logits.device) if sev_weight is not None else None
    l_sev_vec = F.cross_entropy(logits[pos_mask, 1:], labels[pos_mask] - 1, weight=sw, reduction='none')
    w_pos = sample_weights[pos_mask] if sample_weights is not None else None
    return l_bin + _reduce_weighted(l_sev_vec, w_pos)


def focal_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-class Focal Loss (Lin et al., 2017; RQ2-09/10)."""
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    log_pt = log_p.gather(1, labels[:, None]).squeeze(1)
    pt = p.gather(1, labels[:, None]).squeeze(1)
    loss_vec = -((1.0 - pt) ** gamma) * log_pt
    if weight is not None:
        loss_vec = loss_vec * weight[labels]
    return _reduce_weighted(loss_vec, sample_weights)


def soft_ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    sigma: float | torch.Tensor = 0.5,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft ordinal Gaussian target distribution (Diaz & Marathe 2019; M266, C047; RQ2-13 & RQ5-05)."""
    classes = torch.arange(num_classes, device=labels.device, dtype=logits.dtype)[None, :]  # (1, K)
    y = labels[:, None].to(logits.dtype)  # (B, 1)
    if isinstance(sigma, torch.Tensor):
        sig = sigma.to(logits.device, dtype=logits.dtype)[:, None].clamp_min(1e-3)
    else:
        sig = max(float(sigma), 1e-3)
    q = torch.exp(-0.5 * ((classes - y) / sig) ** 2)
    q = q / q.sum(dim=1, keepdim=True)
    log_p = F.log_softmax(logits, dim=1)
    loss_vec = -(q * log_p).sum(dim=1)
    return _reduce_weighted(loss_vec, sample_weights)


def ldam_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margins: torch.Tensor,
    weight: torch.Tensor | None = None,
    scale: float = 30.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """LDAM loss (Cao et al., NeurIPS 2019; RQ2-14)."""
    m = margins.to(logits.device, dtype=logits.dtype)[labels]  # (B,)
    adj = logits.clone()
    adj[torch.arange(len(labels), device=logits.device), labels] -= m
    loss_vec = F.cross_entropy(scale * adj, labels, weight=weight, reduction='none')
    return _reduce_weighted(loss_vec, sample_weights)


def gce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    q: float = 0.7,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generalized Cross-Entropy (Zhang & Sabuncu, NeurIPS 2018; RQ5-08)."""
    p = F.softmax(logits, dim=1)
    py = p.gather(1, labels[:, None]).squeeze(1).clamp(1e-7, 1.0)
    loss_vec = (1.0 - py ** q) / q
    return _reduce_weighted(loss_vec, sample_weights)


# --------------------------------------------------------- Priors & Weights

def logit_adjustment(counts: np.ndarray, tau: float = 1.0) -> torch.Tensor:
    """Menon et al. (2021) log-prior offset: tau * log(pi_k)."""
    prior = np.asarray(counts, dtype=np.float64)
    prior = np.clip(prior / prior.sum(), 1e-8, 1.0)
    return torch.from_numpy((tau * np.log(prior)).astype(np.float32))


def joint_logit_adjustment(joint_counts: dict[str, np.ndarray], tau: float = 1.0) -> dict[str, torch.Tensor]:
    """Per-joint log-prior offset tau * log(pi_{j,k}) for RQ2-17."""
    return {j: logit_adjustment(c, tau=tau) for j, c in joint_counts.items()}


def corn_logit_adjustment(counts: np.ndarray, tau: float = 1.0) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float64)
    k = len(counts)
    offsets = np.zeros(k - 1, dtype=np.float32)
    for j in range(k - 1):
        gt = counts[j + 1:].sum()
        eq = counts[j]
        p = np.clip(gt / max(gt + eq, 1.0), 1e-6, 1.0 - 1e-6)
        offsets[j] = float(tau * np.log(p / (1.0 - p)))
    return torch.from_numpy(offsets)


def effective_number_weights(counts: np.ndarray, beta: float = 0.9999) -> torch.Tensor:
    """Class-Balanced Loss weights (Cui et al., CVPR 2019; RQ2-15): (1 - beta) / (1 - beta^n_c)."""
    c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    eff = (1.0 - np.power(beta, c)) / (1.0 - beta)
    w = 1.0 / eff
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32))


def ldam_margins(counts: np.ndarray, max_m: float = 0.5) -> torch.Tensor:
    """LDAM per-class margins C / n_c^(1/4), normalized so max margin is `max_m` (Cao et al., 2019; RQ2-14)."""
    c = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    m = 1.0 / np.power(c, 0.25)
    m = m * (max_m / m.max())
    return torch.from_numpy(m.astype(np.float32))


# ------------------------------------------------ Supervised Contrastive (RQ1-12/13)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def pooled_and_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, CumulativeLinkWrapper):
        feats = model.backbone(x)
        return feats, model.head(feats)
    feats = model.forward_features(x)
    pooled = model.forward_head(feats, pre_logits=True)
    return pooled, model.forward_head(feats, pre_logits=False)


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    alpha: float | None = None,
    num_classes: int = 5,
) -> torch.Tensor:
    z = F.normalize(z.float(), dim=1)
    n = len(labels)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    same = (labels[:, None] == labels[None, :]) & ~self_mask
    has_pos = same.any(dim=1)
    if not has_pos.any():
        return z.sum() * 0.0
    sim = z @ z.T / temperature
    if alpha is None:
        log_w = torch.zeros_like(sim)
    else:
        dist = (labels[:, None] - labels[None, :]).abs().float() / max(num_classes - 1, 1)
        r = torch.where(same, 1.0, alpha + (1.0 - alpha) * dist)
        log_w = r.log()
    denom = torch.logsumexp((sim + log_w).masked_fill(self_mask, float('-inf')), dim=1)
    per_anchor = -((sim - denom[:, None]) * same).sum(dim=1) / same.sum(dim=1).clamp_min(1)
    return per_anchor[has_pos].mean()


# ------------------------------------------------------------- Inference

@torch.no_grad()
def logits_to_preds(logits: torch.Tensor, head: str = 'ordinal', threshold: float = 0.5) -> torch.Tensor:
    if head in ('softmax', 'cumulative_link'):
        return logits.argmax(dim=1)
    if head == 'softmax_corn':
        k = (logits.shape[1] + 1) // 2
        return logits[:, :k].argmax(dim=1)
    if head == 'ordinal':
        probs = torch.sigmoid(logits)
        return (probs > threshold).sum(dim=1)
    if head == 'corn':
        cond = torch.sigmoid(logits)
        cum = torch.cumprod(cond, dim=1)
        return (cum > threshold).sum(dim=1)
    if head == 'hierarchical':
        p_pos = torch.sigmoid(logits[:, 0])
        sev = logits[:, 1:].argmax(dim=1) + 1
        return torch.where(p_pos > threshold, sev, torch.zeros_like(sev))
    raise ValueError(head)


@torch.no_grad()
def logits_to_probs(logits: torch.Tensor, head: str = 'softmax') -> torch.Tensor:
    """Convert raw model outputs to proper K-class probability simplex P(Y = c | x)."""
    if head == 'softmax':
        return F.softmax(logits, dim=1)
    if head == 'cumulative_link':
        return logits.exp()
    if head == 'softmax_corn':
        k = (logits.shape[1] + 1) // 2
        return F.softmax(logits[:, :k], dim=1)
    if head == 'ordinal':
        gt = torch.sigmoid(logits)  # P(Y > k), shape (B, K-1)
        ones = torch.ones((len(logits), 1), device=logits.device, dtype=logits.dtype)
        zeros = torch.zeros((len(logits), 1), device=logits.device, dtype=logits.dtype)
        gt_ext = torch.cat([ones, gt, zeros], dim=1)
        p = (gt_ext[:, :-1] - gt_ext[:, 1:]).clamp_min(1e-7)
        return p / p.sum(dim=1, keepdim=True)
    if head == 'corn':
        cond = torch.sigmoid(logits)
        gt = torch.cumprod(cond, dim=1)
        ones = torch.ones((len(logits), 1), device=logits.device, dtype=logits.dtype)
        zeros = torch.zeros((len(logits), 1), device=logits.device, dtype=logits.dtype)
        gt_ext = torch.cat([ones, gt, zeros], dim=1)
        p = (gt_ext[:, :-1] - gt_ext[:, 1:]).clamp_min(1e-7)
        return p / p.sum(dim=1, keepdim=True)
    if head == 'hierarchical':
        p_pos = torch.sigmoid(logits[:, 0:1])
        p_sev = F.softmax(logits[:, 1:], dim=1)
        return torch.cat([1.0 - p_pos, p_pos * p_sev], dim=1)
    raise ValueError(head)
