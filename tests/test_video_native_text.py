"""UMT5 layout, tokenizer and attention tests."""
from pathlib import Path

import pytest
import sentencepiece as spm
import torch
from safetensors.torch import save_file

from thenoise.text_encoders.wan_video import WanUMT5, load_wan_text_encoder


CFG = dict(width=16, ff_dim=32, heads=4, layers=2, vocab_size=64)


def small_checkpoint(root: Path, *, missing=False):
    dest = root / "split_files" / "text_encoders"
    dest.mkdir(parents=True)
    training = root / "sentences.txt"
    training.write_text(("A fox walking in snow. Another video of a fox.\n" * 20), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(training), model_prefix=str(root / "sp"),
                                   vocab_size=64, hard_vocab_limit=False,
                                   pad_id=0, eos_id=1, unk_id=2, bos_id=-1,
                                   max_sentence_length=1000)
    with torch.random.fork_rng():
        torch.manual_seed(13)
        model = WanUMT5(**CFG)
    tensors = {key: value.detach().contiguous().half() for key, value in model.state_dict().items()}
    if missing:
        tensors.pop(next(iter(tensors)))
    tensors["spiece_model"] = torch.tensor(list((root / "sp.model").read_bytes()), dtype=torch.uint8)
    save_file(tensors, str(dest / "umt5_xxl_fp16.safetensors"))
    return model


def test_t5_forward_shape_mask_and_no_transformers():
    model = WanUMT5(**CFG).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)
    ids = torch.randint(0, 64, (1, 8))
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        out = model(ids, mask)
    assert out.shape == (1, 8, 16)
    assert torch.isfinite(out).all()
    assert not out[:, -3:].any()


def test_text_encoder_comfy_checkpoint_and_sentencepiece(tmp_path):
    source = small_checkpoint(tmp_path)
    model, tokenizer = load_wan_text_encoder(tmp_path, dtype=torch.float32, _test_config=CFG)
    assert not any(p.is_meta for p in model.parameters())
    assert torch.equal(model.shared.weight, source.shared.weight.half().float())
    ids, mask = tokenizer.encode("A fox", max_length=16)
    assert ids.shape == mask.shape == (1, 16)
    assert ids[0, mask[0]][-1] == 1
    assert not mask[0, -1]
    with torch.no_grad():
        out = model(ids, mask)
    assert out.shape == (1, 16, 16)


def test_text_missing_weights_rejected(tmp_path):
    small_checkpoint(tmp_path, missing=True)
    with pytest.raises(ValueError, match="mismatch"):
        load_wan_text_encoder(tmp_path, _test_config=CFG)
