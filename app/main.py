from fastapi import Depends, FastAPI, HTTPException

from app.llm import ModelTimeout, ModelUnavailable, OllamaClient
from app.models import Product, ProductCard
from app.service import InvalidDraft, create_card


app = FastAPI(title="Product Draft", description="Черновики карточек измерительных приборов")


def get_llm() -> OllamaClient:
    return OllamaClient()


@app.post("/generate", response_model=ProductCard)
async def generate(product: Product, client: OllamaClient = Depends(get_llm)):
    try:
        return await create_card(product, client)
    except ModelTimeout:
        raise HTTPException(504, "Модель не ответила вовремя. Повторите запрос позже.") from None
    except ModelUnavailable:
        raise HTTPException(503, "Модель недоступна. Проверьте Ollama и имя модели.") from None
    except InvalidDraft:
        raise HTTPException(502, "Модель не смогла вернуть корректный черновик за две попытки.") from None
