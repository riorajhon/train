import csv
import re
from collections import defaultdict
import jellyfish
import Levenshtein
import random

def has_consecutive_letters(name, count=3):
    """Check if name has 3 or more consecutive identical letters."""
    for i in range(len(name) - count + 1):
        if len(set(name[i:i+count])) == 1:  # All letters in substring are same
            return True
    return False

def is_valid_name(name):
    """Check if name is valid for training."""
    if not name or len(name) < 2:
        return False
    if not name.isalpha():  # Only alphabetic characters
        return False
    if has_consecutive_letters(name, 3):  # No "aaa" patterns
        return False
    return True

def calculate_phonetic_similarity(original_name: str, variation: str) -> float:
    """Calculate phonetic similarity using multiple algorithms."""
    algorithms = {
        "soundex": lambda x, y: jellyfish.soundex(x) == jellyfish.soundex(y),
        "metaphone": lambda x, y: jellyfish.metaphone(x) == jellyfish.metaphone(y),
        "nysiis": lambda x, y: jellyfish.nysiis(x) == jellyfish.nysiis(y),
    }
    
    random.seed(hash(original_name) % 10000)
    selected_algorithms = random.sample(list(algorithms.keys()), k=min(3, len(algorithms)))
    
    weights = [random.random() for _ in selected_algorithms]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    phonetic_score = sum(algorithms[algo](original_name, variation) * weight
                        for algo, weight in zip(selected_algorithms, normalized_weights))
    return float(phonetic_score)

def calculate_orthographic_similarity(original_name: str, variation: str) -> float:
    """Calculate orthographic similarity using Levenshtein distance."""
    try:
        distance = Levenshtein.distance(original_name, variation)
        max_len = max(len(original_name), len(variation))
        return 1.0 - (distance / max_len)
    except:
        return 0.0

def categorize_similarity(phonetic_score, orthographic_score):
    """Categorize similarity scores into L/M/F buckets."""
    phonetic_boundaries = {"L": (0.80, 1.00), "M": (0.60, 0.79), "F": (0.30, 0.59)}
    orthographic_boundaries = {"L": (0.70, 1.00), "M": (0.50, 0.69), "F": (0.20, 0.49)}
    
    # Check if scores are within valid ranges
    if phonetic_score < 0.30 or orthographic_score < 0.20:
        return None  # Invalid - too low similarity
    
    phonetic_cat = None
    for cat, (min_val, max_val) in phonetic_boundaries.items():
        if min_val <= phonetic_score <= max_val:
            phonetic_cat = cat
            break
    
    orthographic_cat = None
    for cat, (min_val, max_val) in orthographic_boundaries.items():
        if min_val <= orthographic_score <= max_val:
            orthographic_cat = cat
            break
    
    # Return None if either category couldn't be determined
    if phonetic_cat is None or orthographic_cat is None:
        return None
    
    return phonetic_cat + orthographic_cat

def read_csv_file(filename):
    """Read CSV file and extract name variations."""
    name_variations = defaultdict(set)  # Use set to avoid duplicates
    
    print(f"Reading {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            row_count = 0
            
            for row in csv_reader:
                row_count += 1
                if row_count % 10000 == 0:
                    print(f"  Processed {row_count:,} rows...")
                
                if len(row) >= 2:
                    original_name = row[0].strip().strip('"').lower()
                    variations_str = row[1].strip().strip('"')
                    
                    # Validate original name
                    if not is_valid_name(original_name):
                        continue
                    
                    # Split variations and validate each
                    variations = variations_str.split()
                    for variation in variations:
                        variation = variation.strip().lower()
                        if is_valid_name(variation) and variation != original_name:
                            name_variations[original_name].add(variation)
            
            print(f"  Completed {filename}: {row_count:,} rows processed")
    
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    
    return name_variations

def combine_datasets(file1_data, file2_data):
    """Combine datasets and remove duplicates."""
    print("Combining datasets and removing duplicates...")
    
    combined = defaultdict(set)
    
    # Add data from file1
    for name, variations in file1_data.items():
        combined[name].update(variations)
    
    # Add data from file2 (will merge with existing)
    for name, variations in file2_data.items():
        combined[name].update(variations)
    
    # Convert back to regular dict with lists
    result = {}
    for name, variations in combined.items():
        if len(variations) > 0:  # Only keep names with variations
            result[name] = list(variations)
    
    return result

def create_training_dataset(combined_data, max_examples_per_category=5000):
    """Create training dataset with similarity categories."""
    print("Creating training dataset with similarity scores...")
    
    training_data = []
    category_counts = defaultdict(int)
    processed_names = 0
    
    for original_name, variations in combined_data.items():
        processed_names += 1
        if processed_names % 1000 == 0:
            print(f"  Processed {processed_names:,} names...")
        
        for variation in variations:
            # Skip if we have enough examples for all categories
            if all(count >= max_examples_per_category for count in category_counts.values()) and len(category_counts) >= 9:
                break
            
            # Calculate similarity scores
            phonetic_score = calculate_phonetic_similarity(original_name, variation)
            orthographic_score = calculate_orthographic_similarity(original_name, variation)
            category = categorize_similarity(phonetic_score, orthographic_score)
            
            # Skip if category is None (scores too low)
            if category is None:
                continue
            
            # Add to training data if we need more examples of this category
            if category_counts[category] < max_examples_per_category:
                training_data.append({
                    'original_name': original_name,
                    'variation': variation,
                    'similarity_category': category,
                    'phonetic_score': round(phonetic_score, 3),
                    'orthographic_score': round(orthographic_score, 3)
                })
                category_counts[category] += 1
    
    return training_data, category_counts

def save_training_dataset(training_data, filename='training_dataset.csv'):
    """Save training dataset to CSV file."""
    print(f"Saving training dataset to {filename}...")
    
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['original_name', 'variation', 'similarity_category', 'phonetic_score', 'orthographic_score']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in training_data:
            writer.writerow(row)
    
    print(f"✅ Training dataset saved: {len(training_data):,} examples")

def main():
    print("=== PROCESSING LARGE DATASET ===")
    print("This will process ~270,000 data points...")
    
    # Step 1: Read both CSV files
    file1_data = read_csv_file('name_1.csv')
    file2_data = read_csv_file('name_2.csv')
    
    print(f"\nFile 1: {len(file1_data):,} unique names")
    print(f"File 2: {len(file2_data):,} unique names")
    
    # Step 2: Combine and deduplicate
    combined_data = combine_datasets(file1_data, file2_data)
    
    total_variations = sum(len(variations) for variations in combined_data.values())
    print(f"\nCombined: {len(combined_data):,} unique names")
    print(f"Total variations: {total_variations:,}")
    
    # Step 3: Create training dataset
    training_data, category_counts = create_training_dataset(combined_data)
    
    # Step 4: Show statistics
    print(f"\n=== TRAINING DATASET STATISTICS ===")
    print(f"Total training examples: {len(training_data):,}")
    
    print(f"\nDistribution by similarity category:")
    target_categories = ['LL', 'LM', 'LF', 'ML', 'MM', 'MF', 'FL', 'FM', 'FF']
    for category in target_categories:
        count = category_counts.get(category, 0)
        print(f"  {category}: {count:,} examples")
    
    # Step 5: Save dataset
    save_training_dataset(training_data)
    
    print(f"\n✅ Dataset processing complete!")
    print(f"Ready for ML training with {len(training_data):,} examples")

if __name__ == "__main__":
    main()