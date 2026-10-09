import pytest
import torch
from models import MBasedAttention, StructuredMLPForM, TransformerPC, VisionTransformer
from models.baselines import BaselineTransformer, BASELINE_NAMES


def tiny(**kwargs):
    return TransformerPC(d_model=8, n_heads=2, d_ff=16, n_layers=2, dropout=0,
                         mlp_hidden=(12,), **kwargs)


@pytest.mark.parametrize('causal', [False, True])
def test_rowsum_matches_full_matrix_and_gradients(causal):
    torch.manual_seed(1)
    gen = StructuredMLPForM(8, 2, hidden_dims=(12,), dropout=0).double()
    x = torch.randn(2, 5, 8, dtype=torch.double, requires_grad=True)
    context = x.cumsum(1) / torch.arange(1, 6, dtype=x.dtype)[None, :, None] if causal else x.mean(1)
    full = gen(context.reshape(-1, 8))
    expected = full.reshape(*context.shape[:-1], 2, 4, 2, 4).sum(-2)
    actual = gen.head_metrics(x, causal)
    torch.testing.assert_close(actual, expected)
    inputs = [x, *gen.parameters()]
    g1 = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    g2 = torch.autograd.grad(expected.square().sum(), inputs)
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b)


def test_metric_changes_with_input_and_ignores_padding():
    gen = StructuredMLPForM(8, 2, hidden_dims=(12,), dropout=0)
    x = torch.randn(2, 3, 8)
    assert not torch.allclose(gen.head_metrics(x), gen.head_metrics(x + 1))
    padded = torch.cat([x, torch.randn(2, 2, 8) * 100], 1)
    mask = torch.tensor([[1, 1, 1, 0, 0]]).expand(2, -1)
    torch.testing.assert_close(gen.head_metrics(x), gen.head_metrics(padded, padding_mask=mask))


@pytest.mark.parametrize('shared', [False, True])
def test_causal_prefix_matches_full_and_optimizer_contains_all_parameters(shared):
    model = tiny(use_M=True, use_mlp=True, share_mlp=shared).eval()
    optimizer = torch.optim.Adam(model.parameters())
    ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    x = torch.randn(2, 5, 8)
    full = model(x)
    torch.testing.assert_close(full[:, :3], model(x[:, :3]), atol=2e-6, rtol=2e-6)
    assert {id(p) for p in model.parameters()} == ids
    full[..., 0].sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    for layer in model.decoder.layers:
        assert layer.get_attn().get_M().shape == (8, 8)


def test_shared_generator_evaluated_once_per_stack():
    model = tiny(use_M=True, use_mlp=True, share_mlp=True)
    calls = []
    gen = model.decoder.shared_mlp_module
    hook = gen.mlp[0].register_forward_hook(lambda *args: calls.append(1))
    model(torch.randn(2, 5, 8))
    hook.remove()
    assert len(calls) == 1
    assert all(layer.get_attn().mlp_m is gen for layer in model.decoder.layers)


@pytest.mark.parametrize('mode', ['mlp', 'linear_comb', 'bayesian', 'separate_B'])
def test_ablation_modes_backward(mode):
    model = tiny(use_M=True, use_mlp=True, share_mlp=False, off_diag_mode=mode)
    y = model(torch.randn(2, 4, 8))
    loss = y.square().mean() + 1e-4 * model.get_kl_divergence()
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize('use_M', [False, True])
def test_mask_broadcast_and_fully_masked_rows(use_M):
    attn = MBasedAttention(8, 2, dropout=0, use_M=use_M, use_mlp=use_M, mlp_hidden_dims=(8,))
    x = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5, 5, dtype=torch.bool)
    mask[:, :, -1] = False
    mask[:, -1] = False
    y, weights = attn(x, attn_mask=mask)
    assert torch.isfinite(y).all()
    assert weights[..., -1].count_nonzero() == 0
    y.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in attn.parameters() if p.grad is not None)


@pytest.mark.parametrize('baseline', BASELINE_NAMES)
def test_baselines_causal_and_padding(baseline):
    model = BaselineTransformer(baseline, d_model=8, n_heads=4, d_ff=16, n_layers=2, dropout=0).eval()
    x = torch.randn(2, 5, 8)
    torch.testing.assert_close(model(x)[:, :3], model(x[:, :3]), atol=2e-6, rtol=2e-6)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    assert torch.isfinite(model(x, padding_mask=mask)).all()


@pytest.mark.parametrize('shared', [False, True])
def test_vit_parameters_are_eager_and_backward(shared):
    model = VisionTransformer(img_size=8, patch_size=4, num_classes=3, embed_dim=8,
                n_heads=2, depth=2, dropout=0, use_M=True, use_mlp=True,
                share_mlp=shared, mlp_hidden=(12,))
    ids = {id(p) for p in model.parameters()}
    model(torch.randn(2, 3, 8, 8)).sum().backward()
    assert ids == {id(p) for p in model.parameters()}
    assert all(p.grad is not None for p in model.parameters())


def test_static_ablation_is_trainable():
    model = tiny(use_M=True, use_mlp=False)
    model(torch.randn(2, 4, 8))[..., 0].sum().backward()
    assert model.decoder.layers[0].self_attn.M0.grad is not None


@pytest.mark.parametrize('options', [dict(off_diag_mode='linear_comb'), dict(compact_rank=2),
                                    dict(use_layer_bias=True), dict(free_M=True)])
def test_block_ablation_masks_the_final_metric(options):
    gen = StructuredMLPForM(8, 2, hidden_dims=(8,), dropout=0, **options).eval()
    context = torch.randn(2, 8)
    full = gen(context)
    gen.diag_scale = 0
    off = gen(context)
    torch.testing.assert_close(off, full * gen.off_diag_mask)
    assert off.abs().sum() > 0
    gen.diag_scale, gen.off_diag_scale = 1, 0
    torch.testing.assert_close(gen(context), full * gen.diag_mask)
