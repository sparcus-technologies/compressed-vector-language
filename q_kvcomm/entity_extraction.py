"""Entity and fact extraction for hybrid KV communication"""

import re
from typing import List, Optional, Set

import torch


class EntityExtractor:
    """Extract key entities and facts from context"""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self._qa_model = None
        self._qa_tokenizer = None
        self._yake_extractor = None

    def extract_simple(self, context: str, max_tokens: int = 20) -> str:
        """
        Simple extraction: Numbers, capitalized words, key patterns.
        No model needed - works cross-model without dependencies.
        """
        facts = []

        # Extract numbers (ages, dates, times, prices)
        numbers = re.findall(
            r"\b\d+(?:[:.]\d+)?\b(?:\s*(?:AM|PM|years?|days?|dollars?|\$|km/s|%))?\b",
            context,
            re.IGNORECASE,
        )
        facts.extend(numbers[:5])  # Limit to 5 numbers

        # Extract capitalized words (likely names, places)
        # But skip common words
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
        for i, word in enumerate(words):
            cleaned = re.sub(r"[^\w]", "", word)
            if cleaned and cleaned[0].isupper() and cleaned not in common_words:
                # Include some context for multi-word names
                if i + 1 < len(words) and words[i + 1][0].isupper():
                    capitalized.append(f"{cleaned} {words[i + 1]}")
                else:
                    capitalized.append(cleaned)

        # Deduplicate and limit
        seen = set()
        unique_caps = []
        for item in capitalized:
            if item not in seen:
                seen.add(item)
                unique_caps.append(item)
                if len(unique_caps) >= 5:
                    break
        facts.extend(unique_caps)

        # Extract key phrases with "is/are/was/were" (definitions)
        patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|was|were)\s+([^.!?]+)",
            r"([A-Z][a-z]+)\s+(?:works?|lives?|costs?)\s+(?:at|in|as)?\s*([^.!?]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, context)
            for match in matches[:3]:
                if isinstance(match, tuple):
                    # Join the matched groups
                    fact_str = " ".join(match).strip()
                    # Limit length
                    if len(fact_str) < 50:
                        facts.append(fact_str)

        # Join facts with comma separator
        fact_str = ", ".join(facts)

        # Truncate to max tokens (rough estimation: ~4 chars per token)
        max_chars = max_tokens * 4
        if len(fact_str) > max_chars:
            fact_str = fact_str[:max_chars].rsplit(",", 1)[0]

        return fact_str if fact_str else "general information"

    def extract_with_attention(
        self, context: str, query: str, max_tokens: int = 20, device: str = "cuda"
    ) -> str:
        """
        Advanced extraction using model attention to find query-relevant facts.
        Uses the sender model to identify which parts of context matter for the query.
        """
        if self.model is None or self.tokenizer is None:
            return self.extract_simple(context, max_tokens)

        # Combine context and query to see what the model attends to
        combined = f"{context}\n\nQuestion: {query}\nAnswer:"

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    combined, return_tensors="pt", truncation=True, max_length=512
                ).to(device)

                outputs = self.model(**inputs, output_attentions=True, use_cache=False)

                # Get attention from last layer (most semantic)
                attentions = outputs.attentions[-1]  # [batch, heads, seq, seq]

                # Average over heads
                avg_attention = attentions.mean(dim=1)[0]  # [seq, seq]

                # Find which context tokens are most attended to by query tokens
                context_tokens = self.tokenizer.encode(
                    context, add_special_tokens=False
                )
                query_start = len(context_tokens) + 2  # Approximate query position

                if query_start < avg_attention.shape[0]:
                    # Sum attention from query tokens to context tokens
                    query_to_context = avg_attention[
                        query_start:, : len(context_tokens)
                    ]
                    importance = query_to_context.sum(dim=0)  # [context_len]

                    # Get top-k most important tokens
                    top_k = min(max_tokens, len(importance))
                    top_indices = torch.topk(importance, top_k).indices.cpu().tolist()

                    # Extract those tokens
                    important_tokens = [
                        context_tokens[i]
                        for i in sorted(top_indices)
                        if i < len(context_tokens)
                    ]
                    extracted = self.tokenizer.decode(
                        important_tokens, skip_special_tokens=True
                    )

                    # Combine with simple extraction for numbers/names
                    simple_facts = self.extract_simple(context, max_tokens=10)

                    # Merge both approaches
                    combined_facts = (
                        f"{simple_facts}, {extracted}"
                        if simple_facts != "general information"
                        else extracted
                    )

                    # Truncate
                    max_chars = max_tokens * 4
                    if len(combined_facts) > max_chars:
                        combined_facts = combined_facts[:max_chars].rsplit(",", 1)[0]

                    return combined_facts

        except Exception as e:
            # Fallback to simple extraction if attention-based fails
            print(f"Attention extraction failed: {e}, using simple extraction")
            pass

        return self.extract_simple(context, max_tokens)

    def extract_with_qa_model(
        self,
        context: str,
        query: str,
        max_tokens: int = 20,
        device: str = "cuda",
        qa_model_name: str = "deepset/tinyroberta-squad2",
    ) -> str:
        """
        Use a small QA model to extract query-relevant answer span.
        This is more reliable than regex and works cross-model.

        Available lightweight models (sorted by size):
        - "deepset/tinyroberta-squad2" - 82MB, very fast, good accuracy
        - "deepset/roberta-base-squad2" - 496MB, better accuracy
        - "distilbert-base-cased-distilled-squad" - 261MB, balanced

        Default: tinyroberta-squad2 (best speed/accuracy for this use case)
        """
        try:
            # Lazy load QA model (only once)
            if self._qa_model is None:
                from transformers import AutoModelForQuestionAnswering, AutoTokenizer

                print(f"Loading QA model: {qa_model_name}...")
                self._qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
                self._qa_model = AutoModelForQuestionAnswering.from_pretrained(
                    qa_model_name
                ).to(device)
                self._qa_model.eval()

            # Get answer span
            inputs = self._qa_tokenizer(
                query, context, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = self._qa_model(**inputs)

            # Get start and end positions
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1

            # Extract answer tokens
            answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
            answer = self._qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)

            # Also get some context around the answer
            context_window = 10  # tokens before/after
            start_ctx = max(0, answer_start - context_window)
            end_ctx = min(len(inputs["input_ids"][0]), answer_end + context_window)
            context_tokens = inputs["input_ids"][0][start_ctx:end_ctx]
            context_span = self._qa_tokenizer.decode(
                context_tokens, skip_special_tokens=True
            )

            # Combine answer + context, truncate
            extracted = f"{answer}, {context_span}"
            max_chars = max_tokens * 4
            if len(extracted) > max_chars:
                extracted = extracted[:max_chars]

            return extracted

        except Exception as e:
            print(f"QA extraction failed: {e}, falling back to simple")
            return self.extract_simple(context, max_tokens)

    def extract_with_summarization(
        self, context: str, query: str, max_tokens: int = 20, device: str = "cuda"
    ) -> str:
        """
        Use the sender model itself to generate a compressed answer.
        This is the most reliable but requires model inference.
        """
        if self.model is None or self.tokenizer is None:
            return self.extract_simple(context, max_tokens)

        try:
            # Ask the sender model to extract key facts
            prompt = f"Context: {context}\n\nExtract key facts to answer: {query}\nKey facts:"

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            extracted = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
            ).strip()

            return extracted if extracted else self.extract_simple(context, max_tokens)

        except Exception as e:
            print(f"Summarization extraction failed: {e}, falling back to simple")
            return self.extract_simple(context, max_tokens)

    def extract_with_yake(self, context: str, max_tokens: int = 20) -> str:
        """
        Query-independent keyphrase extraction using YAKE.
        Lightweight, no model needed - pure statistical method.

        Perfect for:
        - General context transfer without specific query
        - Extracting important concepts automatically
        - Zero inference overhead
        """
        try:
            # Lazy load YAKE
            if self._yake_extractor is None:
                import yake

                # Parameters tuned for fact extraction
                self._yake_extractor = yake.KeywordExtractor(
                    lan="en",
                    n=3,  # up to 3-gram phrases
                    dedupLim=0.7,  # deduplication threshold
                    top=10,  # extract top 10 keyphrases
                    features=None,
                )

            # Extract keyphrases
            keyphrases = self._yake_extractor.extract_keywords(context)

            # Format: (keyphrase, score) - lower score = more important
            # Sort by score and take most important ones
            sorted_phrases = sorted(keyphrases, key=lambda x: x[1])

            # Combine keyphrases
            facts = [phrase for phrase, score in sorted_phrases[:5]]
            fact_str = ", ".join(facts)

            # Truncate to max tokens
            max_chars = max_tokens * 4
            if len(fact_str) > max_chars:
                fact_str = fact_str[:max_chars].rsplit(",", 1)[0]

            return fact_str if fact_str else "general information"

        except Exception as e:
            print(f"YAKE extraction failed: {e}, falling back to simple")
            return self.extract_simple(context, max_tokens)

    def extract_with_attention_queryless(
        self, context: str, max_tokens: int = 20, device: str = "cuda"
    ) -> str:
        """
        Query-independent extraction using self-attention.
        Finds tokens that are most attended to across the entire context.

        Uses the sender model's attention mechanism to identify salient information
        WITHOUT needing a specific query. Perfect for general context transfer.

        Theory: Tokens that receive high attention from many other tokens are
        semantically important to understanding the context.
        """
        if self.model is None or self.tokenizer is None:
            return self.extract_with_yake(context, max_tokens)

        try:
            # Need to temporarily enable eager attention for attention output
            original_attn_impl = getattr(
                self.model.config, "_attn_implementation", None
            )
            self.model.config._attn_implementation = "eager"

            with torch.no_grad():
                inputs = self.tokenizer(
                    context, return_tensors="pt", truncation=True, max_length=512
                ).to(device)

                outputs = self.model(**inputs, output_attentions=True, use_cache=False)

                # Restore original attention implementation
                if original_attn_impl is not None:
                    self.model.config._attn_implementation = original_attn_impl

                # Check if attention was actually returned
                if not hasattr(outputs, "attentions") or outputs.attentions is None:
                    return self.extract_with_yake(context, max_tokens)

                # Get attention from last layer (most semantic understanding)
                attentions = outputs.attentions[-1]  # [batch, heads, seq, seq]

                # Average over heads
                avg_attention = attentions.mean(dim=1)[
                    0
                ]  # [seq, seq]                # For each token, sum how much attention it receives from all other tokens
                # High score = token is heavily attended to = important
                token_importance = avg_attention.sum(dim=0)  # [seq]

                # Get top-k most important tokens (excluding special tokens)
                input_ids = inputs["input_ids"][0]

                # Create mask for special tokens
                special_token_ids = {
                    self.tokenizer.bos_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.pad_token_id,
                }
                special_mask = torch.tensor(
                    [tid not in special_token_ids for tid in input_ids], device=device
                )

                # Mask out special tokens
                masked_importance = token_importance * special_mask.float()

                # Get top-k important token indices
                k = min(max_tokens, masked_importance.shape[0])
                top_values, top_indices = masked_importance.topk(k)

                # Sort indices by position to maintain order
                top_indices_sorted = top_indices.sort().values

                # Extract tokens and decode
                important_token_ids = input_ids[top_indices_sorted]
                extracted = self.tokenizer.decode(
                    important_token_ids, skip_special_tokens=True
                )

                # Clean up
                extracted = " ".join(extracted.split())  # normalize whitespace

                # Truncate if needed
                max_chars = max_tokens * 4
                if len(extracted) > max_chars:
                    extracted = extracted[:max_chars]

                return (
                    extracted
                    if extracted
                    else self.extract_with_yake(context, max_tokens)
                )

        except Exception as e:
            print(f"Attention extraction failed: {e}, falling back to YAKE")
            return self.extract_with_yake(context, max_tokens)

    def extract(
        self,
        context: str,
        query: str = None,
        max_tokens: int = 20,
        method: str = "simple",
        device: str = "cuda",
    ) -> str:
        """
        Main extraction method - supports both query-dependent and query-independent modes.

        Args:
            context: Context text to extract from
            query: Query text (optional - if None, uses query-independent methods)
            max_tokens: Maximum tokens in extracted facts
            method: "simple", "attention", "qa", "summarization", "yake", "attention_queryless"
            device: Device for model operations

        Returns:
            Extracted facts as string
        """
        # Query-independent methods
        if method == "yake":
            return self.extract_with_yake(context, max_tokens)
        elif method == "attention_queryless":
            return self.extract_with_attention_queryless(context, max_tokens, device)

        # Query-dependent methods (fallback to query-independent if no query)
        if query is None:
            # No query provided - use query-independent method
            if method == "attention":
                return self.extract_with_attention_queryless(
                    context, max_tokens, device
                )
            else:
                return self.extract_with_yake(context, max_tokens)

        # Query provided - use query-dependent methods
        if method == "attention":
            return self.extract_with_attention(context, query, max_tokens, device)
        elif method == "qa":
            return self.extract_with_qa_model(context, query, max_tokens, device)
        elif method == "summarization":
            return self.extract_with_summarization(context, query, max_tokens, device)
        else:  # "simple"
            return self.extract_simple(context, max_tokens)
