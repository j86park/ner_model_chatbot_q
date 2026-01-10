"""
Test script for the KeywordExtractor inference module.
Run this to verify the trained model loads and extracts keywords correctly.
"""

from src.inference import KeywordExtractor


def main():
    print("=" * 60)
    print("    NER KEYWORD EXTRACTOR - TEST SCRIPT")
    print("=" * 60)

    # Load the model
    print("\n[1] Loading model from ./output/my_keyword_model ...")
    extractor = KeywordExtractor(model_path="./output/my_keyword_model")
    print("    Model loaded successfully!")

    # Test query
    test_query = "Show me patents related to car engine cooling systems"

    print(f"\n[2] Test Query:")
    print(f'    "{test_query}"')

    # Extract keywords
    print("\n[3] Extracting keywords...")
    keywords = extractor.extract_keywords(test_query)

    print("\n[4] Extracted Keywords:")
    if keywords:
        for i, kw in enumerate(keywords, 1):
            print(f"    {i}. {kw}")
    else:
        print("    (No keywords extracted)")

    # Additional test cases
    print("\n" + "=" * 60)
    print("    ADDITIONAL TEST CASES")
    print("=" * 60)

    test_cases = [
        "find patents on lithium-ion battery anode materials",
        "hydraulic pump efficiency",
        "search for blockchain consensus mechanisms",
        "what are the patents for variable valve timing systems?",
    ]

    for query in test_cases:
        keywords = extractor.extract_keywords(query)
        print(f'\n  Query: "{query}"')
        print(f"  Keywords: {keywords}")

    print("\n" + "=" * 60)
    print("    TEST COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

