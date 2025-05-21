# Download the pretrained Parakeet model from NVIDIA's NeMo library
import nemo.collections.asr as nemo_asr


asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
print("Pretrained Parakeet model downloaded successfully.")

from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer("avsolatorio/GIST-Embedding-v0", )

# Download the nltk data
import nltk
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/punkt')
except:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/punkt_tab')
except:
    nltk.download('punkt_tab')

print("NLTK data downloaded successfully.")
