import csv
import os

def load_csv_as_dict(file_path):
    """
    Load the CSV file and structure it as a dictionary with level and category name as keys.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Structured dictionary with levels and categories as keys
    """
    # Dictionary to store the result
    prompt_dict = {}
    
    # Track current category
    current_category = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        
        # Skip the first row (header)
        next(csv_reader)
        
        for row in csv_reader:
            # Skip empty rows
            if not row or all(cell.strip() == '' for cell in row):
                continue
                
            # Check if this is the header row with '#' and 'Example'
            if row[0] == '#' and row[1] == 'Example':
                continue
                
            # Check column positions based on the CSV structure
            level1_value = row[0].strip() if len(row) > 0 else ''
            level1_example = row[1].strip() if len(row) > 1 else ''
            level2_value = row[2].strip() if len(row) > 2 else ''
            level2_example = row[3].strip() if len(row) > 3 else ''
            level3_value = row[4].strip() if len(row) > 4 else ''
            level3_example = row[5].strip() if len(row) > 5 else ''
            
            # Check if this is a category row (category rows contain ". " in the name)
            if level1_value and ". " in level1_value:
                current_category = level1_value
                
                # Initialize the categories in the dictionary if they don't exist
                if "Level1_" + current_category not in prompt_dict:
                    prompt_dict["Level1_" + current_category] = []
                    
                if "Level2_" + level2_value not in prompt_dict:
                    prompt_dict["Level2_" + level2_value] = []
                    
                if "Level3_" + level3_value not in prompt_dict:
                    prompt_dict["Level3_" + level3_value] = []
            
            # If this is an example row and we're in a category
            elif current_category and level1_value and level1_value.isdigit():
                # Add examples to the relevant category lists
                if "Level1_" + current_category in prompt_dict:
                    prompt_dict["Level1_" + current_category].append({
                        "id": level1_value,
                        "example": level1_example
                    })
                
                current_level2_category = level2_value.split(". ")[0] + ". " + current_category.split(". ")[1] if ". " in current_category else None
                if current_level2_category and "Level2_" + current_level2_category in prompt_dict:
                    prompt_dict["Level2_" + current_level2_category].append({
                        "id": level2_value,
                        "example": level2_example
                    })
                
                current_level3_category = level3_value.split(". ")[0] + ". " + current_category.split(". ")[1] if ". " in current_category else None
                if current_level3_category and "Level3_" + current_level3_category in prompt_dict:
                    prompt_dict["Level3_" + current_level3_category].append({
                        "id": level3_value,
                        "example": level3_example
                    })
    
    return prompt_dict

def load_structured_prompts(file_path):
    """
    Load the CSV file and structure it into a more hierarchical format.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Hierarchical dictionary with categories and examples
    """
    # Dictionary to store the result
    prompts = {}
    
    # Track current category
    current_category = None
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        
        # Skip the first row (column headers)
        next(csv_reader)
        
        for row in csv_reader:
            # Skip empty rows
            if not row or all(cell.strip() == '' for cell in row):
                continue
                
            # Check if this is the header row with '#' and 'Example'
            if len(row) > 0 and row[0] == '#' and row[1] == 'Example':
                continue
                
            # Check column positions based on the CSV structure
            level1_value = row[0].strip() if len(row) > 0 else ''
            level1_example = row[1].strip() if len(row) > 1 else ''
            level2_value = row[2].strip() if len(row) > 2 else ''
            level2_example = row[3].strip() if len(row) > 3 else ''
            level3_value = row[4].strip() if len(row) > 4 else ''
            level3_example = row[5].strip() if len(row) > 5 else ''
            
            # Check if this is a category row (category rows contain ". " in the name)
            if level1_value and ". " in level1_value:
                current_category = level1_value
                category_name = current_category.split(". ")[1]
                
                # Initialize the category in the dictionary if it doesn't exist
                if category_name not in prompts:
                    prompts[category_name] = {
                        "level1": {
                            "category": current_category,
                            "examples": []
                        },
                        "level2": {
                            "category": level2_value,
                            "examples": []
                        },
                        "level3": {
                            "category": level3_value,
                            "examples": []
                        }
                    }
            
            # If this is an example row and we're in a category
            elif current_category and level1_value and level1_value.isdigit():
                category_name = current_category.split(". ")[1]
                
                # Add examples to the relevant category lists
                if category_name in prompts:
                    if level1_example:
                        prompts[category_name]["level1"]["examples"].append({
                            "id": level1_value,
                            "text": level1_example
                        })
                    
                    if level2_example:
                        prompts[category_name]["level2"]["examples"].append({
                            "id": level2_value,
                            "text": level2_example
                        })
                    
                    if level3_example:
                        prompts[category_name]["level3"]["examples"].append({
                            "id": level3_value,
                            "text": level3_example
                        })
    
    return prompts

# Example usage
if __name__ == "__main__":
    # Define file path
    file_path = "prompts.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
    else:
        # Load the data in two different formats
        flat_dict = load_csv_as_dict(file_path)
        structured_dict = load_structured_prompts(file_path)
        
        # Print all categories to verify we're capturing everything correctly
        print("\nAll Categories Found:")
        all_categories = []
        for key in flat_dict.keys():
            if key.startswith("Level1_"):
                category = key.replace("Level1_", "")
                all_categories.append(category)
        
        for cat in sorted(all_categories):
            print(f"- {cat}")
        print(f"Total categories: {len(all_categories)}")
        
        # Print sample data
        print("\nFlat Dictionary (first 2 keys):")
        for i, (key, value) in enumerate(flat_dict.items()):
            if i < 2:
                print(f"{key}: {value[:2]}")  # Print first 2 examples for each category
            
        print("\nStructured Dictionary (first category):")
        for i, (category, data) in enumerate(structured_dict.items()):
            if i < 1:
                print(f"Category: {category}")
                print(f"  Level 1: {data['level1']['category']}")
                print(f"    First 2 examples: {data['level1']['examples'][:2]}")
                print(f"  Level 2: {data['level2']['category']}")
                print(f"    First 2 examples: {data['level2']['examples'][:2]}")
                print(f"  Level 3: {data['level3']['category']}")
                print(f"    First 2 examples: {data['level3']['examples'][:2]}")
                
        # Save the structured dictionary to a JSON file
        import json
        
        # Create a directory for the JSON files if it doesn't exist
        json_dir = "json_files"
        os.makedirs(json_dir, exist_ok=True)

        # Save the structured dictionary to a JSON file
        json_file_path = os.path.join(json_dir, "structured_prompts.json")
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(structured_dict, json_file, ensure_ascii=False, indent=4)
        