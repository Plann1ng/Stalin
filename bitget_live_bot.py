from __future__ import annotations

"""
Bitget demo/live execution wrapper for the research strategies in research.py.

What this script does
---------------------
- pulls Bitget candles on a fixed interval
- converts them to the same OHLCV shape used by research.py
- evaluates a selected strategy only on fully closed candles
- places market orders on Bitget futures or spot
- tracks one long-only position at a time

What it does NOT do yet
-----------------------
- shorts
- websocket execution
- advanced TP/SL plan orders
- portfolio/multi-symbol orchestration

This is designed as the first clean bridge from your uploaded research code
into Bitget demo trading. Keep it on demo first.
"""

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from research import (  # type: ignore
    BreakoutStrategy,
    MeanReversionStrategy,
    ScalpMomentumStrategy,
    Strategy,
)


BITGET_BASE_URL = "https://api.bitget.com"
DEFAULT_RECV_WINDOW_MS = 5000


@dataclass
class BotConfig:
    symbol: str
    market_type: str  # futures | spot
    product_type: str = "USDT-FUTURES"
    margin_coin: str = "USDT"
    granularity: str = "5m"
    candle_limit: int = 300
    poll_seconds: int = 5
    dry_run: bool = False
    demo: bool = True
    leverage: int = 3
    margin_mode: str = "crossed"
    spot_quote_size: float = 50.0
    futures_size: float = 0.001
    strategy_name: str = "scalp_momentum"


class BitgetAPIError(RuntimeError):
    pass


class BitgetClient:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, demo: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.demo = demo
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "locale": "en-US"})

    def _sign(self, timestamp_ms: str, method: str, request_path: str, query: str = "", body: str = "") -> str:
        payload = f"{timestamp_ms}{method.upper()}{request_path}"
        if query:
            payload += f"?{query}"
        payload += body
        digest = hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _headers(self, method: str, request_path: str, query: str = "", body: str = "") -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-PASSPHRASE": self.passphrase,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-SIGN": self._sign(ts, method, request_path, query, body),
        }
        if self.demo:
            headers["paptrading"] = "1"
        return headers

    def _request(
        self,
        method: str,
        request_path: str,
        params: Optional[Dict[str, Any]] = None,
        body_obj: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        query = urlencode(sorted((params or {}).items())) if params else ""
        body = json.dumps(body_obj, separators=(",", ":")) if body_obj is not None else ""
        url = f"{BITGET_BASE_URL}{request_path}"
        if query:
            url += f"?{query}"
        headers = self._headers(method, request_path, query, body) if auth else None
        resp = self.session.request(method=method.upper(), url=url, headers=headers, data=body if body else None, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != "00000":
            raise BitgetAPIError(f"Bitget error {payload.get('code')}: {payload.get('msg')} | {payload}")
        return payload

    # ---------- public market data ----------
    def get_futures_candles(self, symbol: str, product_type: str, granularity: str, limit: int) -> pd.DataFrame:
        payload = self._request(
            "GET",
            "/api/v2/mix/market/candles",
            params={
                "symbol": symbol,
                "productType": product_type,
                "granularity": granularity,
                "limit": str(limit),
            },
            auth=False,
        )
        return self._candles_to_df(payload["data"])

    def get_spot_candles(self, symbol: str, granularity: str, limit: int) -> pd.DataFrame:
        gran_map = {
            "1m": "1min",
            "3m": "3min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1H": "1h",
        }
        payload = self._request(
            "GET",
            "/api/v2/spot/market/candles",
            params={
                "symbol": symbol,
                "granularity": gran_map.get(granularity, granularity),
                "limit": str(limit),
            },
            auth=False,
        )
        return self._candles_to_df(payload["data"])

    @staticmethod
    def _candles_to_df(raw_rows: Any) -> pd.DataFrame:
        # Bitget returns arrays in reverse chronological order.
        rows = []
        for row in raw_rows:
            if len(row) < 6:
                continue
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True).tz_convert(None),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df.sort_values("timestamp", inplace=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ---------- metadata ----------
    def get_futures_contract_config(self, symbol: str, product_type: str) -> Dict[str, Any]:
        payload = self._request(
            "GET",
            "/api/v2/mix/market/contracts",
            params={"symbol": symbol, "productType": product_type},
            auth=False,
        )
        data = payload["data"]
        if isinstance(data, list):
            for item in data:
                if item.get("symbol") == symbol:
                    return item
            if data:
                return data[0]
        return data

    def get_spot_symbol_info(self, symbol: str) -> Dict[str, Any]:
        payload = self._request(
            "GET",
            "/api/v2/spot/public/symbols",
            params={"symbol": symbol},
            auth=False,
        )
        data = payload["data"]
        if isinstance(data, list):
            for item in data:
                if item.get("symbol") == symbol:
                    return item
            if data:
                return data[0]
        return data

    # ---------- auth/account ----------
    def set_futures_leverage(self, symbol: str, product_type: str, margin_coin: str, leverage: int) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v2/mix/account/set-leverage",
            body_obj={
                "symbol": symbol,
                "productType": product_type,
                "marginCoin": margin_coin,
                "leverage": str(leverage),
            },
            auth=True,
        )

    def get_futures_positions(self, product_type: str, margin_coin: str) -> Any:
        payload = self._request(
            "GET",
            "/api/v2/mix/position/all-position",
            params={"productType": product_type, "marginCoin": margin_coin},
            auth=True,
        )
        return payload["data"]

    def get_spot_assets(self, coin: Optional[str] = None) -> Any:
        params = {"coin": coin} if coin else None
        payload = self._request(
            "GET",
            "/api/v2/spot/account/assets",
            params=params,
            auth=True,
        )
        return payload["data"]

    # ---------- orders ----------
    def place_futures_market_order(
        self,
        symbol: str,
        product_type: str,
        margin_coin: str,
        side: str,
        size: str,
        margin_mode: str,
        reduce_only: str = "NO",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v2/mix/order/place-order",
            body_obj={
                "symbol": symbol,
                "productType": product_type,
                "marginCoin": margin_coin,
                "marginMode": margin_mode,
                "side": side,
                "orderType": "market",
                "size": size,
                "reduceOnly": reduce_only,
                "clientOid": f"demo-{uuid.uuid4().hex[:24]}",
            },
            auth=True,
        )

    def place_spot_market_order(self, symbol: str, side: str, size: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v2/spot/trade/place-order",
            body_obj={
                "symbol": symbol,
                "side": side,
                "orderType": "market",
                "force": "gtc",
                "size": size,
                "clientOid": f"demo-{uuid.uuid4().hex[:24]}",
            },
            auth=True,
        )


def decimal_places_from_step(step: str) -> int:
    d = Decimal(step)
    return max(0, -d.as_tuple().exponent)


def quantize_down(value: float, places: int) -> str:
    q = Decimal("1") if places == 0 else Decimal("1." + ("0" * places))
    return format(Decimal(str(value)).quantize(q, rounding=ROUND_DOWN), f"f")


def build_strategy(name: str) -> Strategy:
    key = name.strip().lower()
    if key == "mean_reversion":
        return MeanReversionStrategy(window=10, std_mult=1.5, max_hold=6)
    if key == "breakout":
        return BreakoutStrategy(lookback=20, ma_window=20, max_hold=6)
    if key == "scalp_momentum":
        return ScalpMomentumStrategy(threshold=0.0005, target=0.0003, stop=0.0007, max_hold=3)
    raise ValueError(f"Unsupported strategy_name={name}")


class LiveStrategyRunner:
    def __init__(self, client: BitgetClient, config: BotConfig):
        self.client = client
        self.config = config
        self.strategy = build_strategy(config.strategy_name)
        self.last_closed_bar_time: Optional[pd.Timestamp] = None
        self.entry_bar_index: Optional[int] = None
        self.entry_price: Optional[float] = None
        self.position_open: bool = False
        self.size_precision: int = 3
        self.price_precision: int = 2
        self.spot_base_coin: Optional[str] = None
        self._bootstrap_market_metadata()

    def _bootstrap_market_metadata(self) -> None:
        if self.config.market_type == "futures":
            info = self.client.get_futures_contract_config(self.config.symbol, self.config.product_type)
            size_place_candidates = [
                info.get("volumePlace"),
                info.get("sizeMultiplier"),
                info.get("minTradeNum"),
            ]
            if info.get("pricePlace") is not None:
                self.price_precision = int(info["pricePlace"])
            vp = info.get("volumePlace")
            if vp is not None:
                self.size_precision = int(vp)
        else:
            info = self.client.get_spot_symbol_info(self.config.symbol)
            self.price_precision = int(info.get("pricePrecision", 2))
            self.size_precision = int(info.get("quantityPrecision", 6))
            self.spot_base_coin = info.get("baseCoin")

    def fetch_bars(self) -> pd.DataFrame:
        if self.config.market_type == "futures":
            return self.client.get_futures_candles(
                self.config.symbol,
                self.config.product_type,
                self.config.granularity,
                self.config.candle_limit,
            )
        return self.client.get_spot_candles(
            self.config.symbol,
            self.config.granularity,
            self.config.candle_limit,
        )

    def current_position_state(self) -> None:
        if self.config.market_type == "futures":
            positions = self.client.get_futures_positions(self.config.product_type, self.config.margin_coin)
            self.position_open = False
            for pos in positions:
                if pos.get("symbol") == self.config.symbol:
                    total = float(pos.get("total", 0) or 0)
                    hold_side = str(pos.get("holdSide", "")).lower()
                    if total > 0 and hold_side in {"long", "buy"}:
                        self.position_open = True
                        return
        else:
            if not self.spot_base_coin:
                self.position_open = False
                return
            assets = self.client.get_spot_assets(self.spot_base_coin)
            self.position_open = False
            for asset in assets:
                if asset.get("coin") == self.spot_base_coin:
                    available = float(asset.get("available", 0) or 0)
                    if available > 0:
                        self.position_open = True
                        return

    def maybe_enter(self, df: pd.DataFrame, i: int) -> None:
        if self.position_open:
            return
        if not self.strategy.should_enter(df, i):
            return
        next_open = float(df["open"].iloc[i + 1])
        if self.config.market_type == "futures":
            size = quantize_down(self.config.futures_size, self.size_precision)
            print(f"ENTER signal @ {df['timestamp'].iloc[i]} -> next open approx {next_open} | size={size}")
            if not self.config.dry_run:
                self.client.place_futures_market_order(
                    symbol=self.config.symbol,
                    product_type=self.config.product_type,
                    margin_coin=self.config.margin_coin,
                    side="buy",
                    size=size,
                    margin_mode=self.config.margin_mode,
                    reduce_only="NO",
                )
        else:
            quote_size = quantize_down(self.config.spot_quote_size, 2)
            print(f"ENTER signal @ {df['timestamp'].iloc[i]} -> next open approx {next_open} | quote_size={quote_size}")
            if not self.config.dry_run:
                self.client.place_spot_market_order(self.config.symbol, "buy", quote_size)
        self.position_open = True
        self.entry_bar_index = i + 1
        self.entry_price = next_open

    def maybe_exit(self, df: pd.DataFrame, i: int) -> None:
        if not self.position_open or self.entry_bar_index is None or self.entry_price is None:
            return
        if not self.strategy.should_exit(df, i, self.entry_bar_index, self.entry_price):
            return
        next_open = float(df["open"].iloc[i + 1])
        if self.config.market_type == "futures":
            size = quantize_down(self.config.futures_size, self.size_precision)
            print(f"EXIT signal @ {df['timestamp'].iloc[i]} -> next open approx {next_open} | size={size}")
            if not self.config.dry_run:
                self.client.place_futures_market_order(
                    symbol=self.config.symbol,
                    product_type=self.config.product_type,
                    margin_coin=self.config.margin_coin,
                    side="sell",
                    size=size,
                    margin_mode=self.config.margin_mode,
                    reduce_only="YES",
                )
        else:
            if not self.spot_base_coin:
                raise RuntimeError("spot_base_coin unavailable")
            assets = self.client.get_spot_assets(self.spot_base_coin)
            sell_size_val = 0.0
            for asset in assets:
                if asset.get("coin") == self.spot_base_coin:
                    sell_size_val = float(asset.get("available", 0) or 0)
                    break
            sell_size = quantize_down(sell_size_val, self.size_precision)
            print(f"EXIT signal @ {df['timestamp'].iloc[i]} -> next open approx {next_open} | sell_size={sell_size}")
            if not self.config.dry_run and float(sell_size) > 0:
                self.client.place_spot_market_order(self.config.symbol, "sell", sell_size)
        self.position_open = False
        self.entry_bar_index = None
        self.entry_price = None

    def on_new_closed_bar(self) -> None:
        df = self.fetch_bars()
        if df.empty or len(df) < self.strategy.max_lookback() + 2:
            print("Not enough bars yet.")
            return

        # Use only fully closed candles for the signal bar. The newest row can still be building,
        # so the second-last row is the decision candle and the last row is the next-bar proxy.
        signal_idx = len(df) - 2
        closed_bar_time = df["timestamp"].iloc[signal_idx]
        if self.last_closed_bar_time is not None and closed_bar_time <= self.last_closed_bar_time:
            return
        self.last_closed_bar_time = closed_bar_time

        self.current_position_state()
        print(
            f"[{pd.Timestamp.utcnow()}] closed_bar={closed_bar_time} symbol={self.config.symbol} "
            f"strategy={self.strategy.name} position_open={self.position_open} close={df['close'].iloc[signal_idx]:.6f}"
        )

        if self.position_open:
            self.maybe_exit(df, signal_idx)
        else:
            self.maybe_enter(df, signal_idx)

    def run_forever(self) -> None:
        if self.config.market_type == "futures" and not self.config.dry_run:
            try:
                self.client.set_futures_leverage(
                    self.config.symbol,
                    self.config.product_type,
                    self.config.margin_coin,
                    self.config.leverage,
                )
                print(f"Leverage set to {self.config.leverage}x")
            except Exception as exc:
                print(f"Warning: could not set leverage automatically: {exc}")

        while True:
            try:
                self.on_new_closed_bar()
            except KeyboardInterrupt:
                print("Stopped by user.")
                break
            except Exception as exc:
                print(f"Loop error: {exc}")
            time.sleep(self.config.poll_seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one of your research.py strategies on Bitget demo/live")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--market-type", choices=["futures", "spot"], default="futures")
    p.add_argument("--product-type", default="USDT-FUTURES")
    p.add_argument("--margin-coin", default="USDT")
    p.add_argument("--granularity", default="5m")
    p.add_argument("--poll-seconds", type=int, default=5)
    p.add_argument("--candle-limit", type=int, default=300)
    p.add_argument("--demo", action="store_true", default=False)
    p.add_argument("--live", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--leverage", type=int, default=3)
    p.add_argument("--margin-mode", default="crossed")
    p.add_argument("--spot-quote-size", type=float, default=50.0, help="For spot market buys: quote-coin size, e.g. 50 USDT")
    p.add_argument("--futures-size", type=float, default=0.001, help="For futures market orders: base size/contracts as accepted by symbol")
    p.add_argument("--strategy", dest="strategy_name", choices=["mean_reversion", "breakout", "scalp_momentum"], default="scalp_momentum")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    api_key = os.getenv("BITGET_API_KEY", "")
    api_secret = os.getenv("BITGET_API_SECRET", "")
    passphrase = os.getenv("BITGET_API_PASSPHRASE", "")
    if not api_key or not api_secret or not passphrase:
        raise SystemExit("Set BITGET_API_KEY, BITGET_API_SECRET, and BITGET_API_PASSPHRASE first.")

    demo = True
    if args.live:
        demo = False
    elif args.demo:
        demo = True

    cfg = BotConfig(
        symbol=args.symbol.upper(),
        market_type=args.market_type,
        product_type=args.product_type.upper(),
        margin_coin=args.margin_coin.upper(),
        granularity=args.granularity,
        candle_limit=args.candle_limit,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
        demo=demo,
        leverage=args.leverage,
        margin_mode=args.margin_mode,
        spot_quote_size=args.spot_quote_size,
        futures_size=args.futures_size,
        strategy_name=args.strategy_name,
    )

    client = BitgetClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase, demo=cfg.demo)
    runner = LiveStrategyRunner(client, cfg)
    runner.run_forever()


if __name__ == "__main__":
    main()
