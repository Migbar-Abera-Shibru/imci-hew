
import json
from pathlib import Path


protocol_path = Path("data/protocol/imci_young_infant.json")
with open(protocol_path, "r", encoding="utf-8") as f:
    protocol = json.load(f)