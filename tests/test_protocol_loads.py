
import json
from pathlib import Path


protocol_path = Path("data/protocol/imci_young_infant.json")
with open(protocol_path, "r", encoding="utf-8") as f:
    protocol = json.load(f)

print(f"Protocol: {protocol['protocol_metadata']['name']}")
print(f"Nodes: {list(protocol['decision_nodes'].keys())}")
print(f"Priority order: {len(protocol['clinical_priority_order'])} classifications")

# verify every criterion sign exists in the sign schema
sign_schema_path = Path("data/sign_schema.json")
with open(sign_schema_path, "r", encoding="utf-8") as f:
    schema = json.load(f)


valid_signs = set()
for category in schema["signs"].values():
    valid_signs.update(category.keys())

missing = []
for node_name, node in protocol["decision_nodes"].items():
    for classification in node.get("classifications", []):
        for sign in classification.get("criteria_any_of", []):
            if sign not in valid_signs:
                missing.append((node_name, classification["name"], sign))

if missing:
    print(f"\n {len(missing)} signs missing from schema:")
    for m in missing:
        print(f"   {m}")
else:
    print("\n All new signs in protocol exist in sign schema")