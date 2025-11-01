"""
Visual demonstration of how entity extraction finds key text.
Shows the step-by-step process of extracting facts from context.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from q_kvcomm import EntityExtractor


def demo_extraction_steps(context: str):
    """Show detailed steps of extraction process"""

    print(f"\n{'='*80}")
    print(f"EXTRACTION DEMO")
    print(f"{'='*80}")
    print(f"\nOriginal Context:")
    print(f"  '{context}'")
    print(f"\n{'-'*80}")

    # Step 1: Extract numbers
    print("\nSTEP 1: Extract Numbers (ages, dates, times, prices)")
    print("-" * 80)
    numbers = re.findall(
        r"\b\d+(?:[:.]\d+)?\b(?:\s*(?:AM|PM|years?|days?|dollars?|\$|km/s|%))?\b",
        context,
        re.IGNORECASE,
    )
    print(f"Pattern: \\b\\d+(?:[:.]\d+)?\\b + optional units (PM, years, $, etc.)")
    print(f"Found: {numbers}")

    # Step 2: Extract capitalized words
    print(f"\n{'-'*80}")
    print("\nSTEP 2: Extract Capitalized Words (names, places, organizations)")
    print("-" * 80)
    common_words = {
        "The",
        "A",
        "An",
        "In",
        "On",
        "At",
        "To",
        "For",
        "Of",
        "And",
        "Or",
        "But",
    }
    words = context.split()
    capitalized = []

    print(f"Process each word:")
    for i, word in enumerate(words):
        cleaned = re.sub(r"[^\w]", "", word)
        if cleaned and cleaned[0].isupper():
            if cleaned in common_words:
                print(f"  '{word}' -> Skipped (common word)")
            else:
                # Check if next word is also capitalized (multi-word names)
                if i + 1 < len(words):
                    next_word = re.sub(r"[^\w]", "", words[i + 1])
                    if (
                        next_word
                        and next_word[0].isupper()
                        and next_word not in common_words
                    ):
                        entity = f"{cleaned} {next_word}"
                        capitalized.append(entity)
                        print(
                            f"  '{word} {words[i+1]}' -> ✓ Multi-word entity: '{entity}'"
                        )
                        continue

                capitalized.append(cleaned)
                print(f"  '{word}' -> ✓ Single entity: '{cleaned}'")

    # Deduplicate
    seen = set()
    unique_caps = []
    for item in capitalized:
        if item not in seen:
            seen.add(item)
            unique_caps.append(item)

    print(f"\nUnique capitalized entities: {unique_caps[:5]}")

    # Step 3: Extract key phrases
    print(f"\n{'-'*80}")
    print("\nSTEP 3: Extract Key Phrases (definitions, relationships)")
    print("-" * 80)

    patterns = [
        (
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|was|were)\s+([^.!?]+)",
            "Pattern: 'Name is/are/was/were <fact>'",
        ),
        (
            r"([A-Z][a-z]+)\s+(?:works?|lives?|costs?)\s+(?:at|in|as)?\s*([^.!?]+)",
            "Pattern: 'Name works/lives/costs at/in/as <place>'",
        ),
    ]

    key_phrases = []
    for pattern, description in patterns:
        print(f"\n{description}")
        matches = re.findall(pattern, context)
        if matches:
            for match in matches[:3]:
                if isinstance(match, tuple):
                    fact_str = " ".join(match).strip()
                    if len(fact_str) < 50:
                        key_phrases.append(fact_str)
                        print(f"  Found: '{fact_str}'")
        else:
            print(f"  (no matches)")

    # Step 4: Combine
    print(f"\n{'-'*80}")
    print("\nSTEP 4: Combine All Extracted Facts")
    print("-" * 80)

    all_facts = numbers[:5] + unique_caps[:5] + key_phrases[:3]
    combined = ", ".join(all_facts)

    print(f"All facts: {all_facts}")
    print(f"\nCombined string:")
    print(f"  '{combined}'")

    # Step 5: Truncate if needed
    max_tokens = 20
    max_chars = max_tokens * 4
    if len(combined) > max_chars:
        truncated = combined[:max_chars].rsplit(",", 1)[0]
        print(f"\nTruncated to ~{max_tokens} tokens:")
        print(f"  '{truncated}'")
        final = truncated
    else:
        final = combined

    print(f"\n{'='*80}")
    print(f"FINAL EXTRACTED FACTS:")
    print(f"{'='*80}")
    print(f"'{final}'")
    print(f"\nEstimated tokens: ~{len(final) // 4}")
    print(f"Original context: {len(context)} chars")
    print(f"Extracted facts: {len(final)} chars")
    print(f"Compression: {len(context) / max(len(final), 1):.1f}x")

    return final


def main():
    print("=" * 80)
    print("HOW ENTITY EXTRACTION WORKS")
    print("=" * 80)
    print("\nThis shows the exact process of extracting key facts from context.")
    print("The extraction is completely rule-based (no ML) so it works cross-model!")

    # Test cases
    test_cases = [
        "Alice is 28 years old and works at Microsoft.",
        "The meeting is scheduled for 2:30 PM in Room 305.",
        "Dr. Sarah Johnson works at Stanford Hospital.",
        "The Tesla Model 3 costs $42,990 and has 358 miles of range.",
        "Python was created by Guido van Rossum in 1991.",
    ]

    for context in test_cases:
        demo_extraction_steps(context)

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(
        """
1. RULE-BASED: Uses regex patterns, no model needed
   - Works with ANY model architecture
   - No training required
   - Deterministic and explainable

2. TARGETS CRITICAL INFO:
   - Numbers: ages, dates, times, prices, IDs
   - Names: capitalized words (people, places, companies)
   - Relationships: "X is Y", "X works at Y"

3. COMPACT: ~20 tokens vs full context (50-100 tokens)
   - Preserves essential facts
   - Reduces by 60-80%

4. WHY IT WORKS:
   - Pure KV cache loses specific facts
   - This text provides the "anchor" facts
   - KV cache provides semantic context
   - Together = accurate + efficient!

ALTERNATIVE: Attention-based extraction
- Uses sender model to identify query-relevant tokens
- More sophisticated but model-dependent
- Simple extraction works well enough for most cases
    """
    )


if __name__ == "__main__":
    main()
