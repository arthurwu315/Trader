"""
Bot A Preflight - 不下單的風控/市況檢查
"""
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / "bots" / "bot_a" / ".env.a_mainnet", override=True)

from bots.bot_a.config_a import get_micro_mvp_config  # noqa: E402
from core.binance_client import BinanceFuturesClient  # noqa: E402
from core.market_data import MarketDataManager  # noqa: E402
from core.market_regime import MarketRegimeDetector  # noqa: E402
from core.risk_manager import RiskManager  # noqa: E402


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/tmp/bot_a_preflight.log'),
            logging.StreamHandler()
        ]
    )


def get_price_1h_ago(market_data, symbol: str) -> float:
    df = market_data.get_klines_df(symbol, "1h", limit=2)
    if len(df) >= 2:
        return float(df["close"].iloc[-2])
    return 0.0


def main():
    setup_logging()
    config = get_micro_mvp_config()

    client = BinanceFuturesClient(
        api_key=config.binance_api_key,
        api_secret=config.binance_api_secret,
        base_url=config.binance_base_url
    )
    market_data = MarketDataManager(client)
    risk_manager = RiskManager(config)
    regime_detector = MarketRegimeDetector(config, market_data, require_structure=True)

    account = client.get_account()
    equity = float(account.get('totalWalletBalance', 0))
    price = float(client.get_ticker_price(config.symbol).get("price", 0))
    price_1h_ago = get_price_1h_ago(market_data, config.symbol)

    risk_manager.set_initial_equity(equity)
    risk_check = risk_manager.comprehensive_risk_check(
        account_balance=equity,
        current_equity=equity,
        start_of_day_equity=equity,
        initial_equity=equity,
        recent_trades=[],
        current_price=price,
        price_1h_ago=price_1h_ago,
    )

    regime_decision = regime_detector.evaluate(config.symbol)

    logging.info("✅ Bot A Preflight 完成")
    logging.info(f"Equity: ${equity:.2f}")
    logging.info(f"RiskCheck: {risk_check}")
    logging.info(f"RegimeDecision: {regime_decision}")


if __name__ == "__main__":
    main()
