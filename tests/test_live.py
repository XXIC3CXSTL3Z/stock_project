from argparse import Namespace

import main as cli_main
from stock_predictor import live


def test_run_live_passes_webhook_env(monkeypatch):
    captured = {}

    def fake_live_signal_once(**kwargs):
        captured["discord_url"] = kwargs.get("discord_url")
        captured["telegram_token"] = kwargs.get("telegram_token")
        captured["telegram_chat_id"] = kwargs.get("telegram_chat_id")
        return []

    def fake_format_recommendations(_):
        return "ok"

    monkeypatch.setenv("SP_DISCORD_WEBHOOK", "https://discord.test/webhook")
    monkeypatch.setenv("SP_TELEGRAM_TOKEN", "TOKEN123")
    monkeypatch.setenv("SP_TELEGRAM_CHAT", "456")
    monkeypatch.setattr(live, "live_signal_once", fake_live_signal_once)
    monkeypatch.setattr(cli_main, "format_recommendations", fake_format_recommendations)

    args = Namespace(
        symbols=["AAPL"],
        period="1mo",
        model="random_forest",
        top=1,
        horizons=[1],
        weighting="markowitz",
        horizon_weights=None,
        crypto=False,
        loop=False,
    )

    cli_main.run_live(args)

    assert captured == {
        "discord_url": "https://discord.test/webhook",
        "telegram_token": "TOKEN123",
        "telegram_chat_id": "456",
    }
