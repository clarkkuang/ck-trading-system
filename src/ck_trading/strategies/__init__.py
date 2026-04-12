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
    low_volatility,
    insider_following,
    short_squeeze,
    sector_rotation,
    earnings_surprise,
    ten_bagger,
)

from ck_trading.strategies.low_volatility import LowVolatilityStrategy  # noqa: F401
from ck_trading.strategies.insider_following import InsiderFollowingStrategy  # noqa: F401
from ck_trading.strategies.short_squeeze import ShortSqueezeStrategy  # noqa: F401
from ck_trading.strategies.sector_rotation import SectorRotationStrategy  # noqa: F401
from ck_trading.strategies.earnings_surprise import EarningsSurpriseStrategy  # noqa: F401
from ck_trading.strategies.ten_bagger import TenBaggerStrategy  # noqa: F401

try:
    from ck_trading.strategies.rl_strategy import RLStrategy  # noqa: F401
except ImportError:
    pass  # stable-baselines3 not installed; RL strategy unavailable
