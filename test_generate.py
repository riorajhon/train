import torch
import torch.nn as nn
import numpy as np

class NameGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, num_layers=1):
        super(NameGeneratorLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Remove dropout for single layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden

class NameVariationGenerator:
    def __init__(self, model_path='test_model.pth', vocab_path='test_vocabulary.pth'):
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
        
        # Load model with matching architecture
        vocab_size = len(self.vocab)
        self.model = NameGeneratorLSTM(vocab_size, embed_dim=16, hidden_dim=32, num_layers=1)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        print("✅ Test name variation generator loaded successfully!")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Vocabulary: {self.vocab}")
    
    def generate_variation(self, original_name, similarity_category, temperature=1.2):
        """Generate a single name variation."""
        # Prepare input
        category_token = self.category_tokens[similarity_category]
        input_text = original_name.lower() + category_token
        
        print(f"Input text: '{input_text}'")
        
        # Convert to indices
        input_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in input_text]
        input_indices = [self.char_to_idx['<START>']] + input_indices
        
        print(f"Input indices: {input_indices}")
        
        # Generate variation
        with torch.no_grad():
            generated = []
            input_tensor = torch.tensor([input_indices])
            hidden = None
            
            # Get initial context from input
            output, hidden = self.model(input_tensor, hidden)
            print(f"Initial output shape: {output.shape}")
            
            # Start generation with START token
            current_input = torch.tensor([[self.char_to_idx['<START>']]])
            
            for step in range(15):  # Generate up to 15 characters
                output, hidden = self.model(current_input, hidden)
                
                # Get last output
                last_output = output[0, -1, :]
                
                # Apply temperature for more diverse sampling
                last_output = last_output / temperature
                probabilities = torch.softmax(last_output, dim=0)
                
                # Show top predictions
                top_probs, top_indices = torch.topk(probabilities, 5)
                print(f"Step {step}: Top chars: {[self.idx_to_char[idx.item()] for idx in top_indices]}")
                
                # Sample next character
                next_char_idx = torch.multinomial(probabilities, 1).item()
                next_char = self.idx_to_char[next_char_idx]
                
                print(f"Step {step}: Selected '{next_char}' (idx: {next_char_idx})")
                
                # Check for end token
                if next_char_idx == self.char_to_idx['<END>']:
                    print("Found END token, stopping generation")
                    break
                
                # Skip special tokens in output
                if next_char_idx not in [self.char_to_idx['<PAD>'], self.char_to_idx['<START>']]:
                    if not next_char.startswith('['):  # Skip category tokens
                        generated.append(next_char)
                        print(f"Added '{next_char}' to output")
                
                # Prepare next input
                current_input = torch.tensor([[next_char_idx]])
            
            result = ''.join(generated).strip()
            print(f"Final result: '{result}'")
            return result

def main():
    print("=== TEST NAME VARIATION GENERATOR ===")
    
    try:
        # Load generator
        generator = NameVariationGenerator()
        
        # Test single generation with debug info
        test_name = "john"
        test_category = "LL"
        
        print(f"\n--- Testing generation for '{test_name}' with category '{test_category}' ---")
        variation = generator.generate_variation(test_name, test_category)
        print(f"\nFinal result: {test_name} → {variation} ({test_category})")
        
        # Test a few more
        test_cases = [("mary", "MM"), ("david", "LF"), ("sarah", "FF")]
        
        for name, category in test_cases:
            print(f"\n--- Testing '{name}' with '{category}' ---")
            variation = generator.generate_variation(name, category)
            print(f"Result: {name} → {variation} ({category})")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()