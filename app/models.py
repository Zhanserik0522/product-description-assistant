from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Text = Field(max_length=200)
    manufacturer: str | None = Field(default=None, max_length=200)
    specs: dict[str, str | None] = Field(default_factory=dict, max_length=50)
    description: str | None = Field(default=None, max_length=12000)

    @field_validator("manufacturer", "description")
    @classmethod
    def clean_optional(cls, value):
        return value.strip() or None if value is not None else None

    @field_validator("specs")
    @classmethod
    def clean_specs(cls, value):
        result = {}
        for key, item in value.items():
            key = key.strip()
            if not key or len(key) > 150:
                raise ValueError("Название характеристики должно содержать 1–150 символов")
            if key in result:
                raise ValueError("Повторяющиеся названия характеристик")
            if item is not None and len(item) > 500:
                raise ValueError("Значение характеристики длиннее 500 символов")
            result[key] = item.strip() or None if item is not None else None
        return result


class GeneratedText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seo_title: Text = Field(max_length=70)
    seo_description: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=800, max_length=1200
    )]
    search_queries: list[Text] = Field(min_length=3, max_length=5)
    application_quote: str | None = Field(max_length=300)


class ProductCard(GeneratedText):
    commercial_block: str
    characteristics: dict[str, str]
    missing_fields: list[str]
    notes: list[str]
