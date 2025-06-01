import os

# List of directories to create
dirs = [
    "data",
    "models",
    "services",
    "api",
    "configs",
    "tests",
    "utils",
    "notebooks"
]

# Files to create at the project root
root_files = [
    "requirements.txt",
    "Dockerfile",
    "README.md"
]

# Files to create inside data/ (exact names as provided)
data_files = [
    "data/Raw",
    "data/processed data",
    "data/embeddings"
]

# Files to create inside models/ (exact names as provided)
model_files = [
    "models/Base models",
    "models/fine-tuned models",
    "models/RAG"
]

# 1) Create all directories (if they don't already exist)
for folder in dirs:
    os.makedirs(folder, exist_ok=True)

# 2) Create the root-level files (empty)
for filepath in root_files:
    with open(filepath, "w"):
        pass

# 3) Create the files inside data/
for filepath in data_files:
    # Ensure the parent directory exists (it should, from step 1)
    parent = os.path.dirname(filepath)
    os.makedirs(parent, exist_ok=True)
    with open(filepath, "w"):
        pass

# 4) Create the files inside models/
for filepath in model_files:
    parent = os.path.dirname(filepath)
    os.makedirs(parent, exist_ok=True)
    with open(filepath, "w"):
        pass

print("✓ All directories and files created successfully.")
