"""Прогнать два примера через API приложения и реальную Ollama."""
import json
import argparse
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", choices=("manometer", "thermometer"))
    args = parser.parse_args()
    folder = Path(__file__).parent / "examples"
    cards = []
    with TestClient(app) as client:
        for name in ((args.product,) if args.product else ("manometer", "thermometer")):
            product = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
            started = time.perf_counter()
            response = client.post("/generate", json=product)
            if response.status_code != 200:
                raise RuntimeError(f"{name}: {response.status_code}, {response.text}")
            result = response.json()
            (folder / f"{name}.result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{name}: {len(result['seo_description'])} chars, "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
    for name in ("manometer", "thermometer"):
        path = folder / f"{name}.result.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            lines = [f"# {result['seo_title']}", "", result["seo_description"],
                     "", "## Характеристики", ""]
            lines += [f"- {key}: {value}" for key, value in result["characteristics"].items()]
            lines += ["", "## Поисковые запросы", "", *[f"- {q}" for q in result["search_queries"]],
                      "", "## Для менеджера", "", result["commercial_block"],
                      "", "## Примечания", "", *[f"- {n}" for n in result["notes"]]]
            cards.append("\n".join(lines))
    (folder / "cards.md").write_text("\n\n---\n\n".join(cards) + "\n", encoding="utf-8")
