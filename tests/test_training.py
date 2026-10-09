import numpy as np
import pytest
import torch
from models import TransformerPC
from train_nlp import NLPTrainer, Seq2SeqHead
from train import Trainer, ReconstructionHead
from utils import save_checkpoint, load_checkpoint, Patchify, MaskGenerator
from evaluation_nlp import compute_all_metrics
from data.datasets_nlp import load_wmt_data


class Tokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    def __len__(self): return 12


def trainer():
    model = TransformerPC(d_model=8, n_heads=2, d_ff=16, n_layers=1,
                          use_M=True, use_mlp=True, share_mlp=True, mlp_hidden=(8,), dropout=0)
    return NLPTrainer(model, Seq2SeqHead(8, 12), Tokenizer(), device='cpu')


def test_source_conditioning_padding_and_embedding_update():
    t = trainer()
    target = torch.tensor([[1, 4, 2], [1, 4, 2]])
    sources = torch.tensor([[1, 5, 2, 0], [1, 6, 2, 0]])
    logits = t._conditioned_logits(sources, target)
    assert not torch.allclose(logits[0], logits[1])
    torch.testing.assert_close(logits, t._conditioned_logits(sources[:, :3], target))
    before = t.embedding.weight.detach().clone()
    optimizer = torch.optim.Adam([*t.model.parameters(), *t.head.parameters(), *t.embedding.parameters()], lr=.01)
    t._compute_loss(logits, target, torch.ones_like(target)).backward()
    optimizer.step()
    assert not torch.equal(before, t.embedding.weight)


def test_complete_checkpoint_roundtrip(tmp_path):
    t = trainer()
    optimizer = torch.optim.Adam([*t.model.parameters(), *t.head.parameters(), *t.embedding.parameters()])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    source, target = torch.tensor([[1, 5, 2]]), torch.tensor([[1, 4, 2]])
    t._conditioned_logits(source, target).sum().backward()
    optimizer.step(); scheduler.step()
    expected = t._conditioned_logits(source, target).detach()
    path = tmp_path / 'model.pt'
    save_checkpoint(t.model, optimizer, 1, 2.5, path,
                    components={'head': t.head, 'embedding': t.embedding}, scheduler=scheduler)
    expected_rand = torch.rand(3)
    other = trainer()
    opt = torch.optim.Adam([*other.model.parameters(), *other.head.parameters(), *other.embedding.parameters()])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
    epoch, loss = load_checkpoint(other.model, opt, path,
                    components={'head': other.head, 'embedding': other.embedding}, scheduler=sched)
    assert (epoch, loss) == (1, 2.5)
    torch.testing.assert_close(torch.rand(3), expected_rand)
    torch.testing.assert_close(other._conditioned_logits(source, target), expected)
    assert sched.state_dict() == scheduler.state_dict()
    assert opt.state_dict()['state']


def test_eos_stops_each_example():
    t = trainer()
    count = [0]
    def logits(source, target):
        result = torch.zeros(2, target.size(1), 12)
        result[0, -1, 2 if count[0] == 0 else 8] = 1
        result[1, -1, 2 if count[0] == 1 else 7] = 1
        count[0] += 1
        return result
    t._conditioned_logits = logits
    assert t.generate(torch.ones(2, 2, dtype=torch.long), max_length=6).tolist() == [[1, 2, 0], [1, 7, 2]]


def test_top_k_bartscore_uses_likelihood_ranking():
    metrics = compute_all_metrics(['a', 'b', 'c'], ['a', 'b', 'c'],
                        [0.2, 0.1, 5.], [-6., -4., -1.], 'translation', top_k=2)
    assert metrics['top50_bartscore_mean'] == -5.


def test_wmt14_parallel_data(tmp_path):
    folder = tmp_path / 'WMT14'; folder.mkdir()
    (folder / 'train.en').write_text('Hello\nWorld\n')
    (folder / 'train.de').write_text('Hallo\nWelt\n')
    assert load_wmt_data(tmp_path) == (['Hello', 'World'], ['Hallo', 'Welt'])
    (folder / 'train.de').write_text('Hallo\n')
    with pytest.raises(ValueError, match='equal'):
        load_wmt_data(tmp_path)


def test_cv_training_updates_patch_projection():
    model = TransformerPC(d_model=8, n_heads=2, d_ff=16, n_layers=1, dropout=0,
                         use_M=True, use_mlp=True, share_mlp=True, mlp_hidden=(8,))
    patcher = Patchify(in_ch=1, img_size=8, patch=2, d_model=8)
    head = ReconstructionHead(8, 4)
    t = Trainer(model, head, patcher, MaskGenerator(mask_ratio=.5, grid_size=4), device='cpu')
    optimizer = torch.optim.Adam([*model.parameters(), *head.parameters(), *patcher.parameters()], lr=.01)
    before = patcher.proj.weight.detach().clone()
    loss, _ = t.train_epoch([(torch.randn(2, 1, 8, 8), None)], optimizer, 1)
    assert np.isfinite(loss)
    assert not torch.equal(before, patcher.proj.weight)


def test_image_metrics_distinguish_distribution_and_pair_distance(monkeypatch):
    import visualization as v
    monkeypatch.setattr(v, '_inception_features', lambda x, device, batch_size=64: x.flatten(1))
    images = torch.randn(8, 1, 2, 2)
    assert v.compute_fid_score(images, images, 'cpu') < 1e-8
    mse, distance = v.compute_per_sample_metrics(images, images + 1, 'cpu')
    torch.testing.assert_close(mse, torch.ones(8))
    torch.testing.assert_close(distance, torch.full((8,), 4.))
    assert v.compute_stats(distance) == (4., 4., 4.)


def test_best_epoch_export(tmp_path):
    from scripts.parse_results import collect_best
    (tmp_path / 'metrics_test.csv').write_text('epoch,val_loss,nll_mean\n1,3.0,3.0\n2,1.0,1.0\n3,nan,0.0\n')
    rows = collect_best(tmp_path)
    assert rows[0]['epoch'] == '2'


def test_training_entrypoint_saves_and_resumes_all_components(tmp_path, monkeypatch):
    import train_nlp as module
    config = dict(dataset='commongen', device='cpu', dev_mode=False,
                  d_model=8, n_heads=2, d_ff=16, n_layers=1,
                  use_M=True, use_mlp=True, share_mlp=True, mlp_hidden=(8,),
                  epochs=2, lr=.001, weight_decay=0, save_every=1,
                  save_dir=str(tmp_path / 'checkpoints'), output_dir=str(tmp_path / 'outputs'))
    batch = dict(source_ids=torch.tensor([[1, 5, 2], [1, 6, 2]]),
                 source_mask=torch.ones(2, 3, dtype=torch.long),
                 target_ids=torch.tensor([[1, 4, 2], [1, 7, 2]]),
                 target_mask=torch.ones(2, 3, dtype=torch.long))
    original_evaluate = module.NLPTrainer.evaluate
    def evaluate(self, loader, epoch=0, compute_metrics=True):
        return original_evaluate(self, loader, epoch, compute_metrics=False)
    batch.update(source_texts=['a', 'b'], target_texts=['a', 'b'])
    monkeypatch.setattr(module.NLPTrainer, 'evaluate', evaluate)
    module.train_nlp([batch], [batch], Tokenizer(), config)
    first = next((tmp_path / 'checkpoints').rglob('checkpoint_epoch_1.pt'))
    saved = torch.load(first, weights_only=True)
    assert set(saved['components']) == {'head', 'embedding'}
    assert saved['optimizer_state_dict']['state']
    final_path = first.with_name('checkpoint_epoch_2.pt')
    original = torch.load(final_path, weights_only=True)
    config['resume'] = str(first)
    module.train_nlp([batch], [batch], Tokenizer(), config)
    resumed = torch.load(first.with_name('checkpoint_epoch_2.pt'), weights_only=True)
    assert resumed['epoch'] == 2
    for name, value in original['model_state_dict'].items():
        torch.testing.assert_close(resumed['model_state_dict'][name], value)
