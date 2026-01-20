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

# Set CPU optimization - leave one core for system
num_cores = os.cpu_count()
cpu_cores_to_use = max(1, num_cores - 1)  # Leave 1 core for system
torch.set_num_threads(cpu_cores_to_use)
print(f"Using {cpu_cores_to_use} CPU cores (leaving 1 for system)")

class NameVariationDataset(Dataset):
    def __init__(self, data, char_to_idx, idx_to_char, max_length=20):
        self.data = data
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.max_length = max_length
        
        # Category tokens
        self.category_tokens = {
            'LL': '[LL]', 'LM': '[LM]', 'LF': '[LF]',
            'ML': '[ML]', 'MM': '[MM]', 'MF': '[MF]',
            'FL': '[FL]', 'FM': '[FM]', 'FF': '[FF]'
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        original_name = row['original_name']
        variation = row['variation']
        category = row['similarity_category']
        
        # Create input: original_name + category_token
        category_token = self.category_tokens[category]
        input_text = original_name + category_token
        
        # Convert to indices
        input_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in input_text]
        target_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in variation]
        
        # Add start token to input, start and end tokens to target
        input_indices = [self.char_to_idx['<START>']] + input_indices
        target_indices = [self.char_to_idx['<START>']] + target_indices + [self.char_to_idx['<END>']]
        
        # Pad sequences
        input_indices = self.pad_sequence(input_indices, self.max_length)
        target_indices = self.pad_sequence(target_indices, self.max_length)
        
        return torch.tensor(input_indices), torch.tensor(target_indices)
    
    def pad_sequence(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.char_to_idx['<PAD>']] * (max_length - len(sequence))

class NameGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_layers=1):
        super(NameGeneratorLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        output = self.fc(lstm_out)
        
        return output, hidden

def create_vocabulary(data):
    """Create character vocabulary from the dataset."""
    all_chars = set()
    
    # Collect all characters from names and variations
    for _, row in data.iterrows():
        all_chars.update(row['original_name'])
        all_chars.update(row['variation'])
    
    # Add special tokens
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    category_tokens = ['[LL]', '[LM]', '[LF]', '[ML]', '[MM]', '[MF]', '[FL]', '[FM]', '[FF]']
    
    # Create vocabulary
    vocab = special_tokens + category_tokens + sorted(list(all_chars))
    
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    return char_to_idx, idx_to_char, vocab

def load_and_prepare_data(csv_file, test_size=0.2):
    """Load and prepare training data."""
    print("Loading dataset...")
    data = pd.read_csv(csv_file)
    
    print(f"Total examples loaded: {len(data):,}")
    
    # Remove rows with NaN values in critical columns
    initial_count = len(data)
    data = data.dropna(subset=['original_name', 'variation'])
    final_count = len(data)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} rows with missing values")
    
    print(f"Total examples after cleaning: {len(data):,}")
    
    # Show category distribution
    category_counts = data['similarity_category'].value_counts()
    print("\nCategory distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count:,}")
    
    # Create vocabulary
    char_to_idx, idx_to_char, vocab = create_vocabulary(data)
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42, 
                                          stratify=data['similarity_category'])
    
    print(f"Training examples: {len(train_data):,}")
    print(f"Validation examples: {len(val_data):,}")
    
    return train_data, val_data, char_to_idx, idx_to_char, vocab

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the LSTM model."""
    # Use the correct padding token index
    pad_idx = train_loader.dataset.char_to_idx['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Batches per epoch: {len(train_loader):,}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass - use teacher forcing
            # Input: original_name + category_token
            # Target: variation (shifted by 1 for next token prediction)
            outputs, _ = model(inputs)
            
            # Calculate loss - predict next token in target sequence
            # Reshape for loss calculation
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = targets.view(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Progress update
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs, _ = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_name_generator.pth')
            print("  ✅ New best model saved!")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        print("-" * 50)
    
    print("Training completed!")

def main():
    print("=== NAME VARIATION GENERATOR TRAINING ===")
    
    # Load and prepare data
    train_data, val_data, char_to_idx, idx_to_char, vocab = load_and_prepare_data('training_dataset.csv')
    
    # Create datasets
    max_length = 25  # Adjust based on your data
    train_dataset = NameVariationDataset(train_data, char_to_idx, idx_to_char, max_length)
    val_dataset = NameVariationDataset(val_data, char_to_idx, idx_to_char, max_length)
    
    # Create data loaders - smaller batch size for CPU
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    vocab_size = len(vocab)
    model = NameGeneratorLSTM(vocab_size, embed_dim=32, hidden_dim=64, num_layers=2)  # 2 layers for better learning
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save vocabulary for inference
    torch.save({
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab': vocab,
        'max_length': max_length
    }, 'vocabulary.pth')
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001)
    
    print("\n✅ Training completed!")
    print("Files saved:")
    print("  - best_name_generator.pth (trained model)")
    print("  - vocabulary.pth (vocabulary and settings)")

if __name__ == "__main__":
    main()