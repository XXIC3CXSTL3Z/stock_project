import importlib
import inspect
from pathlib import Path

MODULES = [
    "stock_predictor.features",
    "stock_predictor.model",
    "stock_predictor.deep",
    "stock_predictor.backtest",
    "stock_predictor.portfolio",
    "stock_predictor.recommend",
    "stock_predictor.fetch",
    "stock_predictor.live",
]


def generate_docs(output: Path) -> None:
    lines = ["# API Reference"]
    for module_name in MODULES:
        mod = importlib.import_module(module_name)
        lines.append(f"\n## {module_name}")
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if obj.__module__ != module_name:
                continue
            doc = inspect.getdoc(obj) or "No docstring."
            lines.append(f"### {name}\n{doc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(lines))


if __name__ == "__main__":
    generate_docs(Path("docs/API.md"))
