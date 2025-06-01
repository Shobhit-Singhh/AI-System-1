import yaml
import json

def load_yaml(file_path):
    """
    Loads a YAML file and returns its contents as a Python dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed contents of the YAML file.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)  # Safe loading prevents code execution
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def save_yaml(data, file_path):
    """
    Saves a Python dictionary to a YAML file.

    Args:
        data (dict): Python dictionary to save.
        file_path (str): Path to the output YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def load_json(file_path):
    """
    Loads a JSON file and returns its contents as a Python dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed contents of the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")


def save_json(data, file_path):
    """
    Saves a Python dictionary to a JSON file.

    Args:
        data (dict): Python dictionary to save.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

