import functools
from typing import Callable

from src.optimizer import LinearMixer, PulayMixer


def get_mixer(name: str, **kwargs) -> Callable[[], object]:
    normalized_name = name.strip().lower()

    if normalized_name == "linear":
        return functools.partial(LinearMixer, alpha=kwargs.get("alpha", 0.5))

    if normalized_name == "pulay":
        return functools.partial(
            PulayMixer,
            alpha=kwargs.get("alpha", 0.5),
            history_size=kwargs.get("history_size", 6),
            regularization=kwargs.get("regularization", 1e-10),
        )

    raise ValueError(f"Unknown mixer '{name}'. Available: linear, pulay")
