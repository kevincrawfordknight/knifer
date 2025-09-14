#!/usr/bin/env python3
import sys, json, re

def clean_text(s: str) -> str:
    s = s.lower()
    # keep only a–z
    return re.sub(r'[^a-z]', '', s)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        text = obj.get("text", "")
    except Exception:
        continue
    cleaned = clean_text(text)
    if cleaned:
        sys.stdout.write(cleaned + "\n")
