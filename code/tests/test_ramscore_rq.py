"""Unit and integration tests for RQ0-RQ5 modules in ramscore."""
from pathlib import Path
import numpy as np
import torch
import pytest

from ramscore.data import NATIVE_SCORES, parse_case, undersample_grade0
from ramscore.metrics import compute_metrics, to_level4
from ramscore.models import (
    build_model,
    corn_logit_adjustment,
    corn_loss,
    effective_number_weights,
    focal_ce_loss,
    gce_loss,
    joint_logit_adjustment,
    ldam_loss,
    ldam_margins,
    logit_adjustment,
    logits_to_preds,
    logits_to_probs,
    ordinal_loss,
    soft_ordinal_loss,
)
from ramscore.rq2 import tune_class_multipliers
from ramscore.rq4 import (
    apply_dirichlet_scaling,
    apply_temperature,
    calibration_metrics,
    conformal_predict_sets,
    fit_dirichlet_scaling,
    fit_temperature,
    risk_coverage_metrics,
)
from ramscore.rq5 import compute_longitudinal_reliability


def test_parse_case_and_metrics():
    c = parse_case('JP_HMCRD_P0001_20210203_6791_L')
    assert (c.centre, c.patient, c.date, c.side) == ('HMCRD', 'HMCRD_P0001', '20210203', 'L')
    y_true = np.array([0, 0, 1, 2, 3, 5])
    y_pred = np.array([0, 1, 1, 2, 5, 3])
    m = compute_metrics(y_true, y_pred)
    assert 0.0 <= m['bacc4'] <= 1.0
    assert -1.0 <= m['qwk'] <= 1.0
    assert list(to_level4(np.array([0, 1, 2, 3, 5]))) == [0, 1, 2, 3, 3]


@pytest.mark.parametrize('head', ['softmax', 'ordinal', 'corn', 'hierarchical', 'cumulative_link', 'softmax_corn'])
def test_heads_and_probs(head):
    net = build_model('resnet18', num_classes=5, head=head, pretrained=False, drop_rate=0.1)
    x = torch.randn(4, 1, 64, 64)
    logits = net(x)
    preds = logits_to_preds(logits, head=head)
    probs = logits_to_probs(logits, head=head)
    assert preds.shape == (4,)
    assert probs.shape == (4, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-4)
    assert (probs >= 0).all()


def test_rq2_losses():
    logits = torch.randn(8, 5, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 3, 4, 0])
    counts = np.array([500, 120, 40, 15, 10])
    sw = torch.rand(8)

    l_focal = focal_ce_loss(logits, labels, gamma=2.0, sample_weights=sw)
    l_soft = soft_ordinal_loss(logits, labels, num_classes=5, sigma=0.5, sample_weights=sw)
    l_ldam = ldam_loss(logits, labels, ldam_margins(counts), weight=effective_number_weights(counts), sample_weights=sw)
    l_gce = gce_loss(logits, labels, q=0.7, sample_weights=sw)

    ord_logits = torch.randn(8, 4, requires_grad=True)
    l_fol = ordinal_loss(ord_logits, labels, num_classes=5, sample_weights=sw, focal_gamma=2.0)
    l_corn = corn_loss(ord_logits, labels, num_classes=5, tau=corn_logit_adjustment(counts), sample_weights=sw)

    for loss in (l_focal, l_soft, l_ldam, l_gce, l_fol, l_corn):
        assert torch.isfinite(loss) and loss.item() > 0


def test_rq2_posthoc_threshold_tuning():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(80, 5))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    gt = rng.choice(NATIVE_SCORES, size=80)
    mult = tune_class_multipliers(probs, gt, NATIVE_SCORES, n_iter=2)
    assert mult.shape == (5,)
    assert (mult > 0).all()


def test_rq4_calibration_and_conformal():
    rng = np.random.default_rng(42)
    n_val, n_test, k = 120, 80, 5
    val_logits = rng.normal(size=(n_val, k))
    val_labels = rng.integers(0, k, size=n_val)
    test_logits = rng.normal(size=(n_test, k))
    test_labels = rng.integers(0, k, size=n_test)

    t_opt = fit_temperature(val_logits, val_labels)
    assert 0.05 <= t_opt <= 20.0
    val_p = apply_temperature(val_logits, t_opt)
    test_p = apply_temperature(test_logits, t_opt)

    W, b = fit_dirichlet_scaling(val_p, val_labels)
    test_p_dir = apply_dirichlet_scaling(test_p, W, b)
    cm = calibration_metrics(test_p_dir, test_labels)
    assert 0.0 <= cm['ece'] <= 1.0
    assert cm['brier'] >= 0.0

    val_joints = rng.choice(['MCP-I', 'PIP-I', 'R'], size=n_val)
    test_joints = rng.choice(['MCP-I', 'PIP-I', 'R'], size=n_test)
    for mode in ('marginal_aps', 'marginal_raps', 'mondrian_grade', 'mondrian_joint_grade'):
        res = conformal_predict_sets(val_p, val_labels, test_p, test_labels, val_joints, test_joints, mode=mode, alpha=0.10)
        assert 0.0 <= res['marginal_coverage'] <= 1.0
        assert 1.0 <= res['mean_set_size'] <= 5.0

    gt_scores = np.asarray(NATIVE_SCORES)[test_labels]
    pred_scores = np.asarray(NATIVE_SCORES)[test_p.argmax(axis=1)]
    rc = risk_coverage_metrics(gt_scores, pred_scores, res['sizes'])
    assert 0.0 <= rc['aurc_bacc4'] <= 1.0


def test_rq5_longitudinal_reliability():
    recs = [
        {'patient': '0001', 'side': 'L', 'joint': 'MCP-I', 'date': '20100101', 'case': '0001_20100101_L', 'score': 2},
        {'patient': '0001', 'side': 'L', 'joint': 'MCP-I', 'date': '20110101', 'case': '0001_20110101_L', 'score': 0},
        {'patient': '0002', 'side': 'R', 'joint': 'PIP-M', 'date': '20100101', 'case': '0002_20100101_R', 'score': 1},
        {'patient': '0002', 'side': 'R', 'joint': 'PIP-M', 'date': '20110101', 'case': '0002_20110101_R', 'score': 2},
    ]
    w = compute_longitudinal_reliability(recs)
    assert w[('0001_20100101_L', 'MCP-I')] < 1.0
    assert w[('0001_20110101_L', 'MCP-I')] < 1.0
    assert w[('0002_20100101_R', 'PIP-M')] == 1.0
    assert w[('0002_20110101_R', 'PIP-M')] == 1.0
