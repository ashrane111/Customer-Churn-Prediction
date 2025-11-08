"""Initialize project directory structure."""
from pathlib import Path

directories = [
    "data/raw", "data/features", "data/processed",
    "models", "logs", "mlruns"
]

for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
    
print("✅ Project directories created!")
