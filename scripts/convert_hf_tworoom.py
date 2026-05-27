"""Convert the Hugging Face TwoRoom weights to the object checkpoint format."""

import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".stable-wm" / "hf_tworooms"
OUT = ROOT / ".stable-wm" / "checkpoints" / "tworoom" / "lewm_object.ckpt"
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from transformers import ViTConfig, ViTModel

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def build_model(cfg):
    enc = cfg["encoder"]
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    }
    vit_cfg = size_configs[enc["size"]].copy()
    vit_cfg["intermediate_size"] = vit_cfg["hidden_size"] * 4
    vit_cfg["image_size"] = enc["image_size"]
    vit_cfg["patch_size"] = enc["patch_size"]

    encoder = ViTModel(
        ViTConfig(**vit_cfg),
        add_pooling_layer=False,
        use_mask_token=enc.get("use_mask_token", False),
    )
    encoder.config.interpolate_pos_encoding = True

    pred = cfg["predictor"]
    act = cfg["action_encoder"]
    projector = cfg["projector"]
    pred_proj = cfg["pred_proj"]

    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(**without_hydra_keys(pred)),
        action_encoder=Embedder(input_dim=act["input_dim"], emb_dim=act["emb_dim"]),
        projector=MLP(
            input_dim=projector["input_dim"],
            output_dim=projector["output_dim"],
            hidden_dim=projector["hidden_dim"],
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=pred_proj["input_dim"],
            output_dim=pred_proj["output_dim"],
            hidden_dim=pred_proj["hidden_dim"],
            norm_fn=nn.BatchNorm1d,
        ),
    )


def without_hydra_keys(values):
    return {key: value for key, value in values.items() if not key.startswith("_")}


def main():
    cfg = json.loads((SRC / "config.json").read_text())
    model = build_model(cfg)
    state_dict = torch.load(SRC / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
