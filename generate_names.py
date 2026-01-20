import torch
import torch.nn as nn
import numpy as np

class NameGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_layers=1):
        super(NameGeneratorLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden

class NameVariationGenerator:
    def __init__(self, model_path='best_name_generator.pth', vocab_path='vocabulary.pth'):
        # Load vocabulary
        vocab_data = torch.load(vocab_path, map_location='cpu')
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = vocab_data['idx_to_char']
        self.vocab = vocab_data['vocab']
        self.max_length = vocab_data['max_length']
        
        # Category tokens
        self.category_tokens = {
            'LL': '[LL]', 'LM': '[LM]', 'LF': '[LF]',
            'ML': '[ML]', 'MM': '[MM]', 'MF': '[MF]',
            'FL': '[FL]', 'FM': '[FM]', 'FF': '[FF]'
        }
        
        # Load model
        vocab_size = len(self.vocab)
        self.model = NameGeneratorLSTM(vocab_size, embed_dim=32, hidden_dim=64, num_layers=1)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        print("✅ Name variation generator loaded successfully!")
        print(f"Vocabulary size: {vocab_size}")
    
    def generate_variation(self, original_name, similarity_category, temperature=1.0, max_attempts=5):
        """Generate a single name variation."""
        for attempt in range(max_attempts):
            # Prepare input
            category_token = self.category_tokens[similarity_category]
            input_text = original_name.lower() + category_token
            
            # Convert to indices
            input_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in input_text]
            input_indices = [self.char_to_idx['<START>']] + input_indices
            
            # Generate variation
            with torch.no_grad():
                generated = []
                input_tensor = torch.tensor([input_indices])
                hidden = None
                
                # Get initial context from input
                output, hidden = self.model(input_tensor, hidden)
                
                # Start generation with START token
                current_input = torch.tensor([[self.char_to_idx['<START>']]])
                
                for step in range(self.max_length - 5):  # Leave room for longer names
                    output, hidden = self.model(current_input, hidden)
                    
                    # Get last output
                    last_output = output[0, -1, :]
                    
                    # Apply temperature for more diverse sampling
                    if temperature > 0:
                        last_output = last_output / temperature
                        probabilities = torch.softmax(last_output, dim=0)
                        
                        # Sample next character (avoid always picking the most likely)
                        next_char_idx = torch.multinomial(probabilities, 1).item()
                    else:
                        # Greedy sampling
                        next_char_idx = torch.argmax(last_output).item()
                    
                    # Check for end token
                    if next_char_idx == self.char_to_idx['<END>']:
                        break
                    
                    # Skip special tokens in output
                    if next_char_idx not in [self.char_to_idx['<PAD>'], self.char_to_idx['<START>']]:
                        next_char = self.idx_to_char[next_char_idx]
                        if not next_char.startswith('['):  # Skip category tokens
                            generated.append(next_char)
                    
                    # Prepare next input
                    current_input = torch.tensor([[next_char_idx]])
                
                result = ''.join(generated).strip()
                
                # Return if we got a valid result that's different from input
                if result and result != original_name.lower():
                    return result
                
                # Try with different temperature
                temperature += 0.2
        
        # Fallback: return a simple variation
        return original_name.lower() + "e"
    
    def generate_distribution(self, original_name, distribution=None):
        """Generate variations according to specified distribution."""
        if distribution is None:
            distribution = {"LL": 3, "MM": 3, "LM": 1, "ML": 1, "FF": 2}
        
        variations = []
        
        for category, count in distribution.items():
            print(f"Generating {count} {category} variations for '{original_name}'...")
            
            for i in range(count):
                variation = self.generate_variation(original_name, category)
                variations.append({
                    'original': original_name,
                    'variation': variation,
                    'category': category
                })
        
        return variations

def main():
    print("=== NAME VARIATION GENERATOR ===")
    
    try:
        # Load generator
        generator = NameVariationGenerator()
        
        # Test generation
        test_names = ["john", "mary", "david", "sarah"]
        
        for name in test_names:
            print(f"\n--- Generating variations for '{name}' ---")
            
            # Generate with default distribution
            variations = generator.generate_distribution(name)
            
            print(f"\nResults for '{name}':")
            for var in variations:
                print(f"  {var['category']}: {var['original']} → {var['variation']}")
        
        # Interactive mode
        print("\n" + "="*50)
        print("INTERACTIVE MODE")
        print("Enter a name and category (e.g., 'john LL') or 'quit' to exit")
        
        while True:
            user_input = input("\nEnter name and category: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            try:
                parts = user_input.split()
                if len(parts) == 2:
                    name, category = parts
                    if category.upper() in generator.category_tokens:
                        variation = generator.generate_variation(name, category.upper())
                        print(f"Generated: {name} → {variation} ({category.upper()})")
                    else:
                        print("Invalid category. Use: LL, LM, LF, ML, MM, MF, FL, FM, FF")
                else:
                    print("Format: <name> <category> (e.g., 'john LL')")
            except Exception as e:
                print(f"Error: {e}")
    
    except FileNotFoundError:
        print("❌ Model files not found. Please train the model first by running:")
        print("   python train_name_generator.py")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    main()