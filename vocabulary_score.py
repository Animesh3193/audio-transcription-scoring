from lexicalrichness import LexicalRichness
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import textstat

LOWER_BOUND = 0
UPPER_BOUND = 10

# Finding the NLTK data files, downloading if not found
try:
    nltk.data.find('corpora/stopwords')
except :
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except :
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/punkt')
except :
    nltk.download('punkt')
try:
    nltk.data.find('corpora/punkt_tab')
except :
    nltk.download('punkt_tab')

def calculate_vocabulary_score(transcribe_output):
    """
    Calculate vocabulary score based on the transcribe output.
    """
    vocab_score = 0
    
    # Fetch Text
    audio_text = transcribe_output[0].text.lower()
    
    # Tokenize the text
    words = word_tokenize(audio_text)
    # Remove punctuation
    words = [word for word in words if word.isalpha()]

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Initialize stop words
    english_stop_words = set(stopwords.words('english'))
    
    # Lemmatize and remove stop words
    lem_word = [lemmatizer.lemmatize(token) for token in words if token not in english_stop_words]
    # Calculate lexical richness
    lem_word_with_stop_words = [lemmatizer.lemmatize(token) for token in words]

    # Calculate lexical richness metrics and mostly used are MTLD and HDD
    # MTLD (Measure of Textual Lexical Diversity)
    # HDD (Hypergeometric Distribution Diversity)
    mtld = 0.0
    hdd = 0.0

    lex = LexicalRichness(lem_word, preprocessor=None, tokenizer=None)

    unique_words = len(set(lem_word))
    hdd_draws_param = max(1, unique_words // 2)

    # making sure that the draws for HDD is less than unique_words
    if hdd_draws_param >= unique_words:
        hdd_draws_param = unique_words - 1
    try:
        mtld = lex.mtld()
        hdd = lex.hdd(draws=hdd_draws_param)
    except Exception as e:
        print(f"Error calculating lexical richness: {e}")
        
        
    # --- Readability Metrics (using textstat) ---
    # Textstat functions typically work on the original, uncleaned text for syllable counting etc.
    flesch_reading_ease = textstat.flesch_reading_ease(audio_text) # Higher score = easier to read
    gunning_fog_index = textstat.gunning_fog(audio_text) # Lower score = easier to read
        
    # ============= Diversity Score calculation ==============
    diversity_score = 0

    # Diversity component (normalized to a certain range)
    # MTLD typically ranges from 0 to 100+, but for scores, we'll cap/scale.
    diversity_score += min(mtld, 50) # Max 50 points for diversity
    diversity_score += min(hdd * 10, 50) # Max 50 points for diversity (HD-D typically 0-1)
    
    # =============== Readability Score calculation ==============
    readability_score = 0
    # Readability component
    # Reward for higher Flesch score, penalize for higher Gunning Fog
    readability_score += min(flesch_reading_ease, 50) # Max 100 points (e.g., Flesch 100 -> 100 pts)
    readability_score += max(0, 5 * (10 - gunning_fog_index)) # Max 100 points (e.g., Gunning 0 -> 100 pts)
    

    # =============== Vocabulary Score calculation ==============
    # Assuming the Lexical Diversity and readability scores contirbute equally to the vocabulary score
    vocab_score = (diversity_score + readability_score) / 2
    # Normalize to a scale of 0-10
    vocab_score = max(LOWER_BOUND, min(vocab_score / UPPER_BOUND, UPPER_BOUND))
    
    return vocab_score