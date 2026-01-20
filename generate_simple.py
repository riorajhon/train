import torch
import torch.nn as nn
import numpy as np

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
        self.decoder = nn.LSTM(embed_dim + 16, hidden_dim, batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, original_seq, category, target_seq=None):
        batch_size = original_seq.size(0)
        
        # Encode original name
        orig_embedded = self.char_embedding(original_seq)
        encoder_out, (hidden, cell) = self.encoder(orig_embedded)
        
        # Get category embedding
        cat_embedded = self.category_embedding(category)
        cat_embedded = cat_embedded.unsqueeze(1)
        
        if target_seq is not None:
            # Training mode
            target_embedded = self.char_embedding(target_seq)
            seq_len = target_embedded.size(1)
            cat_repeated = cat_embedded.repeat(1, seq_len, 1)
            
            decoder_input = torch.cat([target_embedded, cat_repeated], dim=2)
            decoder_out, _ = self.decoder(decoder_input, (hidden, cell))
            
            output = self.output_layer(self.dropout(decoder_out))
            return output
        else:
            # Inference mode
            outputs = []
            current_input = torch.zeros(batch_size, 1, self.char_embedding.embedding_dim).to(original_seq.device)
            decoder_hidden = (hidden, cell)
            
            for _ in range(15):
                decoder_input = torch.cat([current_input, cat_embedded], dim=2)
                decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                
                output = self.output_layer(decoder_out)
                outputs.append(output)
                
                # Use output as next input
                next_char_idx = torch.argmax(output, dim=2)
                current_input = self.char_embedding(next_char_idx)
            
            return torch.cat(outputs, dim=1)

class NameVariationGenerator:
    def __init__(self, model_path='simple_name_generator.pth', vocab_path='simple_vocabulary.pth'):
        # Load vocabulary
        vocab_data = torch.load(vocab_path, map_location='cpu')
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = vocab_data['idx_to_char']
        self.vocab = vocab_data['vocab']
        self.max_length = vocab_data['max_length']
        
        # Category mappings
        self.category_to_idx = {
            'LL': 0, 'LM': 1, 'LF': 2, 'ML': 3, 'MM': 4, 
            'MF': 5, 'FL': 6, 'FM': 7, 'FF': 8
        }
        
        # Load model
        vocab_size = len(self.vocab)
        self.model = SimpleNameGenerator(vocab_size, category_size=9, embed_dim=64, hidden_dim=128)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        print("✅ Simple name generator loaded!")
        print(f"Vocabulary size: {vocab_size}")
    
    def generate_variation(self, original_name, similarity_category, temperature=1.0):
        """Generate a name variation."""
        # Prepare input
        name_lower = original_name.lower()
        name_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in name_lower]
        
        # Pad to max length
        if len(name_indices) > self.max_length:
            name_indices = name_indices[:self.max_length]
        else:
            name_indices = name_indices + [self.char_to_idx['<PAD>']] * (self.max_length - len(name_indices))
        
        # Convert to tensors
        name_tensor = torch.tensor([name_indices])
        category_tensor = torch.tensor([self.category_to_idx[similarity_category]])
        
        # Generate
        with torch.no_grad():
            output = self.model(name_tensor, category_tensor)
            
            # Convert output to characters
            generated_chars = []
            for i in range(output.size(1)):
                char_probs = torch.softmax(output[0, i] / temperature, dim=0)
                
                # Sample character
                if temperature > 0:
                    char_idx = torch.multinomial(char_probs, 1).item()
                else:
                    char_idx = torch.argmax(char_probs).item()
                
                char = self.idx_to_char[char_idx]
                
                # Stop at padding or end
                if char in ['<PAD>', '<END>']:
                    break
                
                if char not in ['<START>', '<UNK>']:
                    generated_chars.append(char)
            
            result = ''.join(generated_chars).strip()
            return result if result else original_name.lower() + "e"
    
    def generate_multiple(self, original_name, similarity_category, count=3):
        """Generate multiple variations."""
        variations = []
        for i in range(count):
            temp = 0.8 + (i * 0.2)  # Vary temperature for diversity
            variation = self.generate_variation(original_name, similarity_category, temp)
            variations.append(variation)
        return variations

def main():
    print("=== SIMPLE NAME VARIATION GENERATOR ===")
    
    try:
        generator = NameVariationGenerator()
        
        # Test cases
        test_cases = [
            ("john", "LL"), ("john", "MM"), ("john", "LF"),
            ("mary", "LL"), ("mary", "FF"), ("mary", "MM"),
            ("david", "LL"), ("david", "LF"), ("david", "MM"),
            ("sarah", "LL"), ("sarah", "FF"), ("sarah", "MM")
        ]
        
        for name, category in test_cases:
            variations = generator.generate_multiple(name, category, 3)
            print(f"{name} ({category}): {variations}")
        
        # Interactive mode
        print("\n" + "="*50)
        print("INTERACTIVE MODE")
        print("Enter 'name category' (e.g., 'john LL') or 'quit'")
        
        while True:
            user_input = input("\nEnter: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            try:
                parts = user_input.split()
                if len(parts) == 2:
                    name, category = parts
                    if category.upper() in generator.category_to_idx:
                        variations = generator.generate_multiple(name, category.upper(), 5)
                        print(f"{name} → {variations}")
                    else:
                        print("Invalid category. Use: LL, LM, LF, ML, MM, MF, FL, FM, FF")
                else:
                    print("Format: <name> <category>")
            except Exception as e:
                print(f"Error: {e}")
    
    except FileNotFoundError:
        print("❌ Model files not found. Run train_simple_generator.py first!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()