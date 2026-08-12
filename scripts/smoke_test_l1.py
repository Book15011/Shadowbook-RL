import json
import sys
import time
from typing import Literal

import requests
from pydantic import BaseModel, Field, ValidationError


class L1Signal(BaseModel):
    timestamp_ms: int
    regime: Literal["risk_on", "risk_off", "neutral", "high_volatility"]
    risk_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    urgency_multiplier: float = Field(ge=0.5, le=2.0)
    rationale: str


def main() -> int:
    prompt = (
        "Return only JSON matching this schema: "
        "{timestamp_ms:int, regime:enum[risk_on,risk_off,neutral,high_volatility], "
        "risk_score:float[-1,1], confidence:float[0,1], urgency_multiplier:float[0.5,2], "
        "rationale:str}."
    )

    payload = {
        "model": "qwen2.5:14b-instruct-q4_K_M",
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"FAIL: request error: {exc}")
        return 1

    body = resp.json()
    raw = body.get("response", "")

    try:
        parsed = json.loads(raw)
        validated = L1Signal(**parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        print("FAIL: schema validation failed")
        print(raw)
        print(exc)
        return 1

    if validated.timestamp_ms <= 0:
        validated.timestamp_ms = int(time.time() * 1000)

    print("PASS")
    print(validated.model_dump_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
