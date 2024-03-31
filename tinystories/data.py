import torch
from torch.utils.data import Dataset, DataLoader
from sentencepiece import SentencePieceProcessor

class TinyStoriesDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: SentencePieceProcessor, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self.load_data(file_path)
    
    def load_data(self, file_path: str):
        with open(file_path, "r") as f:
            file = f.read()
        
        return file.split("<|endoftext|>")
    
    def __getitem__(self, index):
        story = self.data[index]
        tokens = self.tokenizer.encode(story)
        if len(tokens) > 512:
            tokens = tokens[:512]
        else:
            tokens = tokens + [0] * (512 - len(tokens))
        tokens = torch.tensor(tokens)
        return tokens
    
    def __len__(self):
        return len(self.data)
