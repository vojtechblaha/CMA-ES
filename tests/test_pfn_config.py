import tempfile

import torch

from pfn_cmaes.stubs.decision_models import PFNBackboneConfig, RegularPFNBackbone

cfg = PFNBackboneConfig(
    hidden_dim=96,
    num_heads=4,
    num_context_layers=3,
    num_query_layers=1,
    ff_multiplier=2,
    dropout=0.0,
    activation="relu",
    use_type_embeddings=False,
)

model = RegularPFNBackbone(context_dim=17, query_dim=9, config=cfg)

ckpt = {
    "model_state_dict": model.state_dict(),
    "context_dim": 17,
    "query_dim": 9,
    "backbone_config": {
        "hidden_dim": 96,
        "num_heads": 4,
        "num_context_layers": 3,
        "num_query_layers": 1,
        "ff_multiplier": 2,
        "dropout": 0.0,
        "activation": "relu",
        "use_type_embeddings": False,
    },
}

path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
torch.save(ckpt, path)
