"""Model construction for the public EvoEP training pipeline."""

from .evoep import EvoEP


MODELS = {"evoep": EvoEP}


def build_model(cfg, text_feature_dim):
    try:
        factory = MODELS[cfg.model.name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {cfg.model.name}; choices: {sorted(MODELS)}"
        ) from exc
    return factory(cfg, text_feature_dim)
