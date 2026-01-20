import torch
import pandas as pd
from train_name_generator import *

print("=== QUICK TRAINING TEST ===")

# Load a small subset for testing
data = pd.read_csv('training_dataset.csv')
data = data.dropna(subset=['original_name', 'variation'])
small_data = data.head(1000)  # Use only 1000 examples for quick test

print(f"Using {len(small_data)} examples for quick test")

# Create vocabulary
char_to_idx, idx_to_char, vocab = create_vocabulary(small_data)
print(f"Vocabulary size: {len(vocab)}")

# Create dataset
max_length = 20
dataset = NameVariationDataset(small_data, char_to_idx, idx_to_char, max_length)

# Create data loader
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Create model
vocab_size = len(vocab)
model = NameGeneratorLSTM(vocab_size, embed_dim=16, hidden_dim=32, num_layers=1)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Quick training (just a few batches)
criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()
for batch_idx, (inputs, targets) in enumerate(train_loader):
    if batch_idx >= 10:  # Only train on 10 batches
        break
        
    optimizer.zero_grad()
    outputs, _ = model(inputs)
    
    outputs_flat = outputs.view(-1, outputs.size(-1))
    targets_flat = targets.view(-1)
    
    loss = criterion(outputs_flat, targets_flat)
    loss.backward()
    optimizer.step()
    
    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

print("Quick test completed!")

# Save for testing
torch.save({
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab': vocab,
    'max_length': max_length
}, 'test_vocabulary.pth')

torch.save(model.state_dict(), 'test_model.pth')
print("Test files saved: test_model.pth, test_vocabulary.pth")