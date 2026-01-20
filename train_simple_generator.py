import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import os
import time
from sklearn.model_selection import train_test_split

# Set CPU optimization
num_cores = os.cpu_count()
cpu_cores_to_use = max(1, num_cores - 1)
torch.set_num_threads(cpu_cores_to_use)
print(f"Using {cpu_cores_to_use} CPU cores")

class SimpleNameDataset(Dataset):
    def __init__(self, data, char_to_idx, idx_to_char, max_length=20):
        self.data = data
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.max_length = max_length
        
        # Category embeddings
        self.category_to_idx = {
            'LL': 0, 'LM': 1, 'LF': 2, 'ML': 3, 'MM': 4, 
            'MF': 5, 'FL': 6, 'FM': 7, 'FF': 8
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        original_name = row['original_name'].lower()
        variation = row['variation'].lower()
        category = row['similarity_category']
        
        # Convert names to indices
        orig_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in original_name]
        var_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in variation]
        
        # Pad sequences
        orig_indices = self.pad_sequence(orig_indices, self.max_length)
        var_indices = self.pad_sequence(var_indices, self.max_length)
        
        # Category index
        cat_idx = self.category_to_idx[category]
        
        return (torch.tensor(orig_indices), 
                torch.tensor(cat_idx), 
                torch.tensor(var_indices))
    
    def pad_sequence(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.char_to_idx['<PAD>']] * (max_length - len(sequence))

class SimpleNameGenerator(nn.Module):
    def __init__(self, vocab_size, category_size=9, embed_dim=64, hidden_dim=128):
        super(SimpleNameGenerator, self).__init__()
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Category embedding
        self.category_embedding = nn.Embedding(category_size, 16)
        
        # Encoder for original name
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Decoder for variation
        self.decoder = nn.LSTM(embed_dim + 16, hidden_dim, batch_first=True)  # +16 for category
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, original_seq, category, target_seq=None):
        batch_size = original_seq.size(0)
        
        # Encode original name
        orig_embedded = self.char_embedding(original_seq)
        encoder_out, (hidden, cell) = self.encoder(orig_embedded)
        
        # Get category embedding
        cat_embedded = self.category_embedding(category)  # [batch, 16]
        cat_embedded = cat_embedded.unsqueeze(1)  # [batch, 1, 16]
        
        if target_seq is not None:
            # Training mode - teacher forcing
            target_embedded = self.char_embedding(target_seq)
            
            # Concatenate category to each timestep
            seq_len = target_embedded.size(1)
            cat_repeated = cat_embedded.repeat(1, seq_len, 1)  # [batch, seq_len, 16]
            
            decoder_input = torch.cat([target_embedded, cat_repeated], dim=2)
            decoder_out, _ = self.decoder(decoder_input, (hidden, cell))
            
            output = self.output_layer(self.dropout(decoder_out))
            return output
        else:
            # Inference mode - generate step by step
            outputs = []
            current_input = torch.zeros(batch_size, 1, self.char_embedding.embedding_dim).to(original_seq.device)
            decoder_hidden = (hidden, cell)
            
            for _ in range(20):  # Max generation length
                # Add category info
                decoder_input = torch.cat([current_input, cat_embedded], dim=2)
                decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                
                output = self.output_layer(decoder_out)
                outputs.append(output)
                
                # Use output as next input
                next_char_idx = torch.argmax(output, dim=2)
                current_input = self.char_embedding(next_char_idx)
            
            return torch.cat(outputs, dim=1)

def create_vocabulary(data):
    """Create character vocabulary from the dataset."""
    all_chars = set()
    
    # Collect all characters from names and variations
    for _, row in data.iterrows():
        all_chars.update(row['original_name'].lower())
        all_chars.update(row['variation'].lower())
    
    # Add special tokens
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    
    # Create vocabulary
    vocab = special_tokens + sorted(list(all_chars))
    
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    return char_to_idx, idx_to_char, vocab

def train_model(model, train_loader, val_loader, num_epochs=20):
    """Train the model."""
    pad_idx = train_loader.dataset.char_to_idx['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (orig_seq, category, target_seq) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(orig_seq, category, target_seq)
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for orig_seq, category, target_seq in val_loader:
                output = model(orig_seq, category, target_seq)
                loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'simple_name_generator.pth')
            print("  ✅ Best model saved!")
        
        print("-" * 40)

def main():
    print("=== SIMPLE NAME GENERATOR TRAINING ===")
    
    # Load data
    data = pd.read_csv('training_dataset.csv')
    data = data.dropna(subset=['original_name', 'variation'])
    
    print(f"Total examples: {len(data):,}")
    
    # Create vocabulary
    char_to_idx, idx_to_char, vocab = create_vocabulary(data)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Characters: {vocab}")
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create datasets
    max_length = 15
    train_dataset = SimpleNameDataset(train_data, char_to_idx, idx_to_char, max_length)
    val_dataset = SimpleNameDataset(val_data, char_to_idx, idx_to_char, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = SimpleNameGenerator(len(vocab), category_size=9, embed_dim=64, hidden_dim=128)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save vocabulary
    torch.save({
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab': vocab,
        'max_length': max_length
    }, 'simple_vocabulary.pth')
    
    # Train
    train_model(model, train_loader, val_loader, num_epochs=15)
    
    print("\n✅ Training completed!")
    print("Files saved:")
    print("  - simple_name_generator.pth")
    print("  - simple_vocabulary.pth")

if __name__ == "__main__":
    main()