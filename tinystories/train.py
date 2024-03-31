import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tinystories.data import TinyStoriesDataset
from tinystories.model import Transformer
from sentencepiece import SentencePieceProcessor

import wandb

model = Transformer(
    vocab_size=512,
    embed_size=128,
    num_layers=4,
    heads=8,
    dropout=0.1,
    forward_expansion=256,
    device="mps:0"
)
tokenizer = SentencePieceProcessor(model_file="./tinystories_tokenizer.model")

model.load_state_dict(torch.load("./transformer_0_2000.pt"))

model.to("mps:0")

dataset = TinyStoriesDataset(
    tokenizer=tokenizer,
    max_len=512,
    file_path="./data/TinyStories-train.txt"
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3

wandb.init(project="tinystories", config={
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": num_epochs
})

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        input_tokens, target_tokens = batch[:, :-1], batch[:, 1:]
        input_tokens = input_tokens.to("mps:0")
        target_tokens = target_tokens.to("mps:0")

        out = model(input_tokens)
        loss = F.cross_entropy(out.contiguous().view(-1, out.size(-1)), target_tokens.contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item()}")
            wandb.log({"loss": loss.item()})
        
        if batch_idx % 1000 == 0:
            torch.save(model.state_dict(), f"transformer_{epoch}_{batch_idx}.pt")

            # out is logits. now go to target_tokens and get the argmax.
            out_tokens = torch.argmax(out, dim=2)
            pred_text = tokenizer.DecodePieces(out_tokens.tolist())
            target_text = tokenizer.DecodePieces(target_tokens.tolist())
            wandb.log({"pred_text": pred_text, "target_text": target_text})

