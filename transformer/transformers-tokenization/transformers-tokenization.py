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
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            idx = len(self.word_to_id)
            self.word_to_id[token] = idx
            self.id_to_word[idx]=token

        unique_sets = set()
        for text in texts:
            unique_sets.update(text.lower().split())

        for word in sorted(unique_sets):
            if word not in self.word_to_id:
                idx = len(self.word_to_id)
                self.word_to_id[word]=idx
                self.id_to_word[idx]=word

        self.vocab_size = len(self.word_to_id)
                
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        unk_id = self.word_to_id[self.unk_token]
        words = text.lower().split()

        return [self.word_to_id.get(word, unk_id) for word in words]
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = [self.id_to_word.get(i, "<UNK>") for i in ids]

        return " ".join(words)
        pass
