import asyncio
import html
import os
import time
from typing import Callable, Iterable, Optional

import requests

from .recommend import format_recommendations, generate_recommendations


def _format_alert_message(message: str, title: str = "Stock Signals") -> tuple[str, str]:
    """Pretty-print the table as a code block for Discord and Telegram."""
    text = message.strip()
    discord_msg = f"**{title}**\n```\n{text}\n```"
    telegram_msg = f"<b>{html.escape(title)}</b>\n<pre>{html.escape(text)}</pre>"
    return discord_msg, telegram_msg


def send_alert(message: str, discord_url: Optional[str] = None, telegram_token: Optional[str] = None, telegram_chat_id: Optional[str] = None) -> None:
    """Send a message to Discord and/or Telegram if webhooks are provided."""
    discord_msg, telegram_msg = _format_alert_message(message)
    try:
        if discord_url:
            requests.post(discord_url, json={"content": discord_msg}, timeout=5)
    except Exception:
        pass
    try:
        if telegram_token and telegram_chat_id:
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            requests.post(
                url,
                data={
                    "chat_id": telegram_chat_id,
                    "text": telegram_msg,
                    "parse_mode": "HTML",
                },
                timeout=5,
            )
    except Exception:
        pass


def live_signal_once(
    tickers: Iterable[str],
    period: str = "6mo",
    crypto: bool = False,
    discord_url: Optional[str] = None,
    telegram_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    **kwargs,
):
    """Generate signals once and optionally push alerts."""
    preds = generate_recommendations(tickers=tickers, period=period, crypto=crypto, **kwargs)
    message = format_recommendations(preds)
    send_alert(message, discord_url=discord_url, telegram_token=telegram_token, telegram_chat_id=telegram_chat_id)
    return preds


def live_signal_loop(
    tickers: Iterable[str],
    period: str = "6mo",
    crypto: bool = False,
    interval: int = 900,
    on_update: Optional[Callable] = None,
    **kwargs,
):
    """Poll predictions on an interval for live monitoring."""
    discord_url = os.getenv("SP_DISCORD_WEBHOOK")
    telegram_token = os.getenv("SP_TELEGRAM_TOKEN")
    telegram_chat_id = os.getenv("SP_TELEGRAM_CHAT")
    while True:
        preds = live_signal_once(
            tickers=tickers,
            period=period,
            crypto=crypto,
            discord_url=discord_url,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
            **kwargs,
        )
        if on_update:
            on_update(preds)
        time.sleep(max(60, interval))


async def stream_crypto_prices(symbols: Iterable[str], on_message: Callable[[dict], None], stream_url: str = "wss://stream.binance.com:9443/ws") -> None:
    """
    Lightweight real-time stream for crypto via WebSocket (Binance public streams).
    Requires the `websockets` package; no-op if unavailable.
    """
    try:
        import websockets  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Install `websockets` to use streaming mode.") from exc

    streams = "/".join(f"{sym.lower()}usdt@trade" for sym in symbols)
    url = f"{stream_url}/{streams}"
    async with websockets.connect(url) as ws:
        async for msg in ws:
            on_message(msg)
