"""Value investing strategies.

Importing this package triggers registration of all built-in strategies
via the :func:`ck_trading.strategies.registry.register` decorator.
"""

# Import strategy modules to trigger @register side-effects
from ck_trading.strategies import (  # noqa: F401
    composite_value,
    dcf_intrinsic,
    dual_momentum,
    earnings_momentum,
    garp,
    graham_defensive,
    magic_formula,
    mean_reversion,
    momentum,
    piotroski_f_score,
    quality_factor,
    risk_parity,
    trend_following,
)
