import json
import re

from pydantic import ValidationError

from app.llm import OllamaClient
from app.models import GeneratedText, Product, ProductCard
from app.prompts import SYSTEM_PROMPT


class InvalidDraft(Exception):
    pass


def numbers(text: str) -> set[str]:
    return {value.replace(",", ".") for value in re.findall(r"\d+(?:[.,]\d+)?", text)}


def validate_draft(raw: str, product: Product) -> GeneratedText:
    draft = GeneratedText.model_validate_json(raw)
    if len({q.casefold() for q in draft.search_queries}) != len(draft.search_queries):
        raise ValueError("Поисковые запросы должны различаться")
    source = json.dumps(product.model_dump(), ensure_ascii=False)
    output = " ".join([draft.seo_title, draft.seo_description,
                       *draft.search_queries, draft.application_quote or ""])
    unknown = numbers(output) - numbers(source)
    if unknown:
        raise ValueError(f"Во входных данных нет чисел: {', '.join(sorted(unknown))}")
    standards = r"\b(ГОСТ|ISO|DIN|IEC|EN)(?=\b|[а-я])"
    unsupported = set(re.findall(standards, output)) - set(re.findall(standards, source))
    if unsupported:
        raise ValueError("В источнике нет подтверждения стандартов: " + ", ".join(sorted(unsupported)))
    if draft.application_quote is not None:
        if not draft.application_quote.strip() or draft.application_quote not in (product.description or ""):
            raise ValueError("application_quote должна быть точной цитатой из description или null")
    return draft


async def create_card(product: Product, client: OllamaClient) -> ProductCard:
    source = product.model_dump_json()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source},
    ]
    # Одна попытка исправления: не зацикливаем запросы к модели.
    for attempt in range(2):
        raw = await client.generate(messages)
        try:
            draft = validate_draft(raw, product)
            break
        except (ValidationError, ValueError) as exc:
            if attempt == 1:
                raise InvalidDraft from exc
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                 f"Исправь JSON. Ошибки проверки: {exc}. "
                 "Текст описания должен содержать 800–1200 символов. "
                 "Все факты бери только из исходного JSON. Верни полный результат."},
            ])

    characteristics = {key: value or "данных нет" for key, value in product.specs.items()}
    missing = [f"specs.{key}" for key, value in product.specs.items() if value is None]
    for field in ("manufacturer", "description"):
        if getattr(product, field) is None:
            missing.append(field)
    if not product.specs:
        missing.append("specs")
        characteristics["Характеристики"] = "данных нет"
    notes = ["Черновик: перед публикацией проверьте текст по паспорту прибора."]
    if missing:
        notes.append("Есть пропуски во входных данных: " + ", ".join(missing))
    notes.append("Недостающие характеристики нужно уточнить по паспорту или у поставщика.")
    commercial = "Задача применения не указана. Запросите у заказчика назначение прибора."
    if draft.application_quote:
        commercial = (f"Задача заказчика по исходному описанию: «{draft.application_quote}» "
                      "Перед предложением прибора согласуйте условия эксплуатации "
                      "и требования к подключению.")
    return ProductCard(**draft.model_dump(), commercial_block=commercial, characteristics=characteristics,
                       missing_fields=missing, notes=notes)
