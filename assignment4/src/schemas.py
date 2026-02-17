from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class Money(BaseModel):
    currency: str = Field(..., description="ISO 4217 currency code, e.g. USD")
    amount: float = Field(..., ge=0)


class Person(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None


class ActionItem(BaseModel):
    description: str
    owner: Literal["sender", "recipient", "unknown"] = "unknown"
    due_date: Optional[date] = None


class ExtractedEmail(BaseModel):
    """
    Schema for structured extraction from a raw email message.
    """

    # Metadata
    subject: Optional[str] = None
    sent_at: Optional[datetime] = None
    from_: Optional[Person] = Field(default=None, alias="from")
    to: list[Person] = Field(default_factory=list)
    cc: list[Person] = Field(default_factory=list)

    # Content understanding
    summary: str
    intent: Literal[
        "request",
        "complaint",
        "support",
        "scheduling",
        "payment",
        "information",
        "other",
    ]
    urgency: Literal["low", "medium", "high"]
    sentiment: Literal["positive", "neutral", "negative", "mixed"]

    # Entities / structured fields
    mentioned_dates: list[date] = Field(default_factory=list)
    mentioned_money: list[Money] = Field(default_factory=list)
    mentioned_links: list[HttpUrl] = Field(default_factory=list)

    # What to do next
    action_items: list[ActionItem] = Field(default_factory=list)

