import torch

from model.models_mae import mae_vit_small_patch4


def test_small_mae_shapes_and_mask_count():
    torch.manual_seed(7)
    model = mae_vit_small_patch4()
    model.eval()
    images = torch.rand(1, 3, 32, 128)
    confidence = torch.linspace(0, 1, 256).unsqueeze(0)
    with torch.no_grad():
        prediction, mask, reconstruction_target = model(images, confidence)

    assert prediction.shape == (1, 256, 48)
    assert mask.shape == (1, 256)
    # Final MSINet-B uses [0.4, 0.3, 0.2, 0.1]. Per-stratum integer
    # truncation masks 66/256 patches.
    assert int(mask.sum().item()) == 66
    assert [int(mask[0, start:start + 64].sum()) for start in (192, 128, 64, 0)] == [26, 20, 13, 7]
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(reconstruction_target).all()


def test_small_mae_checkpoint_key_families():
    model = mae_vit_small_patch4()
    keys = set(model.state_dict())
    assert "patch_embed.proj.weight" in keys
    assert "pos_embed" in keys
    assert "mask_token" in keys
    assert "decoder_pred.weight" in keys
