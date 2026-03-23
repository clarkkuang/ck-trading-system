"""Strategy registry for dynamic strategy discovery and instantiation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ck_trading.strategies.base import Strategy

_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator that registers a strategy in the global registry.

    Uses the class's ``name`` property (via a throwaway instance) as the key.
    """
    # Instantiate to read the name property; all strategies have cheap __init__
    instance = cls()
    _REGISTRY[instance.name] = cls
    return cls


def get_all_strategies() -> dict[str, type]:
    """Return ``{display_name: class}`` for every registered strategy."""
    return dict(_REGISTRY)


def get_strategy(name: str, **kwargs) -> "Strategy":
    """Instantiate a registered strategy by its display name.

    Raises ``KeyError`` if *name* is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Strategy {name!r} not found. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)
