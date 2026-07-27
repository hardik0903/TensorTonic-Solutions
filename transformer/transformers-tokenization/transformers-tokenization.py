import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        # Step 1: append word_to_id with special token first
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            idx = len(self.word_to_id)
            self.word_to_id[token] = idx
            self.id_to_word[idx]=token

        # Step 2: create unique text in lower and indiviual from texts
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)

        # Step 3: append the sorted unique words from set into word_to_id
        for words in sorted(unique_words):
            if words not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[words] = idx
                self.id_to_word[idx] = words

        self.vocab_size = len(self.word_to_id)
                
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE

        # Step 1: lower and split the give string that needs to be encoded
        words = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]

        # Step 2: check if word exists in word_to_id if yes then append its id else appends unk_id
        return [self.word_to_id.get(word, unk_id) for word in words]
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        # Step 1: see if that particular id maps to a word in id_to_word if yes append the word else append "<UNK>"
        words = [self.id_to_word.get(i, "<UNK>") for i in  ids]

        # Step 2: convert the words to string
        return " ".join(words)
        # YOUR CODE HERE
        pass
