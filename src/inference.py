from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


class KeywordExtractor:
    """
    NER-based keyword extractor using a fine-tuned transformer model.
    Uses the transformers pipeline with aggregation to merge sub-tokens.
    """

    def __init__(self, model_path="./output/my_keyword_model"):
        """
        Load the trained model and tokenizer.

        Args:
            model_path: Path to the directory containing model artifacts.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        # Create pipeline with simple aggregation to merge B-KEY/I-KEY tokens
        self.ner_pipeline = pipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )

    def extract_keywords(self, text):
        """
        Extract keyword phrases from input text.

        Args:
            text: Input string to extract keywords from.

        Returns:
            List of extracted keyword strings.
        """
        # Run NER pipeline
        entities = self.ner_pipeline(text)

        # Filter for KEY entities and extract the word/phrase
        keywords = []
        for entity in entities:
            # entity_group will be "KEY" after aggregation (B-KEY/I-KEY merged)
            if "KEY" in entity["entity_group"]:
                word = entity["word"].strip()
                # Clean up any tokenizer artifacts (like ## from BERT)
                word = word.replace(" ##", "").replace("##", "")
                if word:
                    keywords.append(word)

        return keywords

