"""
Synthetic training data generator for Stage 2 fine-tuning.

Purpose:
Generate Amharic sign-extraction examples from the IMCI protocol.
Each example maps a natural Amharic clinical description
to the structured JSON signs the engine consumes.

Input:  IMCI protocol JSON + sign schema
Output: JSONL file with {instruction, input, output} triplets
"""

from __future__ import annotations


import json
from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "data" / "protocol" / "imci_young_infant.json"
SIGN_SCHEMA_PATH = PROJECT_ROOT / "data" / "sign_schema.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "extraction_train.jsonl"

random.seed(3407)

# Amharic phrasing for each sign key
# Multiple variants per sign so the model learns robust extraction

SIGN_PHRASES = {
    # Feeding
    "not_able_to_feed_at_all": [
        "ልጁ ምንም ማጥባት አይችልም",
        "ልጁ ጨርሶ አይጠባም",
        "ልጁ ወተት መውሰድ አይችልም",
    ],
    "not_feeding_well": [
        "ልጁ በደንብ አይጠባም",
        "ልጁ በበቂ ሁኔታ አይመገብም",
        "የልጁ መመገብ ቀንሷል",
    ],
    "breastfeeding_less_than_8_times_24h": [
        "ልጁ በ24 ሰዓት ከ8 ጊዜ በታች ይጠባል",
        "ልጁ በቀን ውስጥ ጥቂት ጊዜ ብቻ ይጠባል",
    ],
    "not_well_attached": [
        "ልጁ በጡት በደንብ አልተያዘም",
        "የልጁ የጡት አያያዝ ትክክል አይደለም",
    ],
    "not_sucking_effectively": [
        "ልጁ በደንብ አይጠባም",
        "የልጁ መጥባት ደካማ ነው",
    ],
    "receives_other_foods": [
        "ልጁ ከወተት ውጭ ሌላ ምግብ ይወስዳል",
        "ልጁ ተጨማሪ ምግብ ይመገባል",
    ],

    # Neurological
    "convulsions": [
        "ልጁ መንቀጥቀጥ አለው",
        "ልጁ የመንቀጥቀጥ ጥቃት አጋጥሞታል",
        "ልጁ ተንቀጠቀጠ",
    ],
    "movement_only_when_stimulated": [
        "ልጁ ሲነካ ብቻ ይንቀሳቀሳል",
        "ልጁ ሲነሳሳ ብቻ ይንቀሳቀሳል ከዚያም ይቆማል",
    ],
    "no_movement_at_all": [
        "ልጁ ጨርሶ አይንቀሳቀስም",
        "ልጁ ምንም እንቅስቃሴ የለውም",
    ],
    "restless_irritable": [
        "ልጁ ይተነፍሳል እና ይበሳጫል",
        "ልጁ ተንቀሳቃሽ እና ተበሳጭ ነው",
    ],

    # Respiratory
    "severe_chest_indrawing": [
        "ልጁ ጥልቅ የደረት መጎተት አለው",
        "የልጁ ደረት ወደ ውስጥ ይጎተታል",
    ],
    "fast_breathing_60_plus_under_7_days": [
        "ልጁ ከ7 ቀናት በታች ሲሆን በደቂቃ ከ60 በላይ ይተነፍሳል",
        "አዲስ የተወለደው ልጅ በደቂቃ ከ60 ጊዜ በላይ ይተነፍሳል",
    ],
    "fast_breathing_60_plus_7_to_59_days": [
        "ልጁ በደቂቃ ከ60 ጊዜ በላይ ይተነፍሳል",
        "የልጁ መተንፈስ ፈጣን ነው በደቂቃ 60 ጊዜ ይተነፍሳል",
    ],
    "stridor": [
        "ልጁ ሲተነፍስ የሚሰማ ድምፅ አለው",
        "ልጁ የስትራይደር ድምፅ አለው",
    ],

    # Temperature
    "high_temp_38_plus": [
        "የልጁ የሰውነት ሙቀት 38 ዲግሪ ወይም ከዚያ በላይ ነው",
        "ልጁ ከፍተኛ ትኩሳት አለው 38.5 ዲግሪ",
        "የልጁ ሙቀት 39 ዲግሪ ነው",
    ],
    "low_temp_below_35_5": [
        "የልጁ የሰውነት ሙቀት ከ35.5 ዲግሪ በታች ነው",
        "ልጁ በጣም ቀዝቃዛ ነው 35 ዲግሪ",
    ],

    # Skin / Umbilicus
    "umbilicus_red_or_pus": [
        "የልጁ እምብርት ቀይ ነው",
        "የልጁ እምብርት አረማመጥ ያፈሳል",
        "የልጁ ሆድ ጉበት ቀይ ነው",
    ],
    "skin_pustules": [
        "በልጁ ቆዳ ላይ አረማመጥ አለ",
        "የልጁ ቆዳ አረማመጥ ያሳያል",
    ],

    # Hydration
    "sunken_eyes": [
        "የልጁ ዓይኖች ወደ ውስጥ ገብተዋል",
        "የልጁ ዓይኖች ጎድተዋል",
    ],
    "skin_pinch_very_slow": [
        "የልጁ ቆዳ ሲነጠፍ በጣም ቀስ ብሎ ይመለሳል",
        "የልጁ ሆድ ቆዳ ከ2 ሰከንድ በላይ ይወስዳል ሲመለስ",
    ],
    "skin_pinch_slow": [
        "የልጁ ቆዳ ሲነጠፍ ቀስ ብሎ ይመለሳል",
        "የልጁ ቆዳ መመለስ ቀርፋፋ ነው",
    ],

    # Jaundice
    "palms_soles_yellow": [
        "የልጁ መዳፍ እና እግር ቢጫ ነው",
        "የልጁ እጆች እና እግሮች ቢጫ ሆነዋል",
    ],
    "jaundice_present": [
        "ልጁ ቢጫ ቀለም አለው",
        "የልጁ ቆዳ እና ዓይን ቢጫ ነው",
    ],
    "jaundice_increasing": [
        "የልጁ ቢጫነት እየባሰ ነው",
        "የልጁ ቢጫ ቀለም አልቀነሰም",
    ],
    "jaundice_beyond_3_weeks": [
        "የልጁ ቢጫነት ከ3 ሳምንት በላይ ቆይቷል",
        "ልጁ ከ3 ሳምንት በላይ ቢጫ ነው",
    ],

    # Weight
    "weight_under_2kg": [
        "የልጁ ክብደት ከ2 ኪሎ ግራም በታች ነው",
        "ልጁ ከ2 ኪሎ በታች ነው",
    ],
    "weight_for_age_below_minus_2z": [
        "የልጁ ክብደት ከዕድሜው ጋር ሲነጻጸር በጣም ዝቅተኛ ነው",
        "የልጁ ክብደት ከ-2 Z በታች ነው",
    ],

    # Mouth
    "thrush": [
        "በልጁ አፍ ውስጥ ነጭ ንጣፍ አለ",
        "የልጁ አፍ ነጭ ቁስሎች አሉት",
    ],

    # HIV
    "mother_hiv_positive": [
        "እናቱ ኤች አይ ቪ ፖዘቲቭ ናት",
        "የልጁ እናት ኤች አይ ቪ አለባት",
    ],
    "infant_virological_positive": [
        "የልጁ የቫይሮሎጂ ምርመራ ፖዘቲቭ ነው",
        "ልጁ ኤች አይ ቪ ፖዘቲቭ ነው",
    ],
    "hiv_test_not_done": [
        "የኤች አይ ቪ ምርመራ አልተደረገም",
        "እናቱም ልጁም አልተመረመሩም",
    ],

    # Duration
    "diarrhoea_stopped": [
        "የልጁ ተቅማጥ ቆሟል",
        "ልጁ ከተቅማጥ ተረጋግቷል",
    ],
}


# amharic instruction template

INSTRUCTION_AMHARIC = (
    "ከሚከተለው የጤና ጽሑፍ የህክምና ምልክቶችን አውጣ። "
    "መልሱን በJSON ብቻ ስጥ። ምልክት ከሌለ ባዶ JSON ስጥ።"
)

INSTRUCTION_EN = (
    "Extract the clinical signs from the following health text. "
    "Return only JSON. If no signs are present, return an empty JSON object."
)

# generation logic

def _build_sample(sign_keys: list[str]) -> dict:
    """Build one training example from a list of sign keys."""
    phrases = []
    for key in sign_keys:
        variants = SIGN_PHRASES.get(key)
        if not variants:
            continue
        phrases.append(random.choice(variants))

    # Shuffle phrase order so the model doesn't learn positional patterns
    random.shuffle(phrases)
    input_text = " ".join(phrases)

    output_obj = {key: True for key in sign_keys}
    output_text = json.dumps(output_obj, ensure_ascii=False)

    return {
        "instruction": INSTRUCTION_AMHARIC,
        "input": input_text,
        "output": output_text,
        "signs": sign_keys,  # metadata for debugging
    }

def generate(num_samples: int = 2500) -> list[dict]:
    """Generate num_samples synthetic sign-extraction examples."""
    all_signs = list(SIGN_PHRASES.keys())
    samples: list[dict] = []

    for _ in range(num_samples):
        # Distribution:
        #   40% single sign  (easy)
        #   40% two signs    (medium)
        #   20% three signs  (harder)
        roll = random.random()
        if roll < 0.4:
            k = 1
        elif roll < 0.8:
            k = 2
        else:
            k = 3

        chosen = random.sample(all_signs, k=k)
        samples.append(_build_sample(chosen))

    # Add a few empty examples so model learns "no signs" case
    for _ in range(100):
        samples.append({
            "instruction": INSTRUCTION_AMHARIC,
            "input": random.choice([
                "ልጁ ጤናማ ነው",
                "ልጁ ምንም ችግር የለውም",
                "የልጁ ሁኔታ ጥሩ ነው",
            ]),
            "output": "{}",
            "signs": [],
        })

    random.shuffle(samples)
    return samples


def main() -> None:
    samples = generate(num_samples=2500)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} examples to {OUTPUT_PATH}")
    print(f"Sample:\n{json.dumps(samples[0], indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()