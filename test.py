import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Ensure variables are set
# os.environ["HF_HOME"] = os.getenv("HF_HOME", os.path.join(os.getcwd(), "huggingface_cache"))
# os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", os.path.join(os.getcwd(), "huggingface_cache", "datasets"))

# Verify the values
print("HF_HOME:", os.getenv("HF_HOME"))
print("HF_DATASETS_CACHE:", os.getenv("HF_DATASETS_CACHE"))
