import re
import nltk
from nltk.tokenize import word_tokenize
import language_tool_python

LOWER_BOUND = 0
UPPER_BOUND = 10

# Download NLTK resources and assign the language tool en-US for grammar checking
try:
    try:
        nltk.data.find('corpora/punkt')
    except :
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/punkt_tab')
    except :
        nltk.download('punkt_tab')
    lang_tool = language_tool_python.LanguageTool('en-US')
except Exception as e:
    print(f"Error downloading NLTK resources or initializing language tool: {e}")
    lang_tool = None
    
    
def calculate_grammar_score(transcribe_output):
    """
    Calculate grammar score based on the transcribe output.
    """
    grammar_score = 0
    
    # Fetch Text
    audio_text = transcribe_output[0].text
    
    # Removing filler words from the text to avoid classifying them as correct grammar
    filler_word_pattern=r'\b(um|uh|ah|err|hmm|like|you know|i mean)\b'

    clean_audio_text = re.sub(filler_word_pattern, '', audio_text)
    # To Remove extra spaces
    clean_audio_text = re.sub(r'\s+', ' ', clean_audio_text).strip()
    
    # Tokenize the text
    words = word_tokenize(clean_audio_text)
    
    # Define a simple penalty system for demonstration
    # You would need to refine this based on specific rule IDs for your use case
    penalty_map = {
        'GRAMMAR_ERROR': 5,      # General, more severe issues
        'TYPOGRAPHICAL_ERROR': 1, # Spelling/punctuation issues
        'STYLE_ERROR': 2,        # Stylistic suggestions
        'UNCATEGORIZED': 3       # Default for others
    }
    
    # Check for grammatical errors using language_tool_python
    if lang_tool:
        
        # Check for grammatical errors
        match_texts = lang_tool.check(audio_text)
        num_errors = len(match_texts)
        
        total_penalty = 0
        for match in match_texts:
            rule_id = match.ruleId
            message = match.message
            # Attempt to categorize error type
            error_type = 'UNCATEGORIZED'
            if 'AGREEMENT' in rule_id or 'SVA' in rule_id or 'TENSE' in rule_id or 'PRONOUN' in rule_id:
                error_type = 'GRAMMAR_ERROR'
            elif 'COMMA' in rule_id or 'PUNCTUATION' in rule_id or 'SPELLING' in rule_id:
                error_type = 'TYPOGRAPHICAL_ERROR'
            elif 'REDUNDANCY' in rule_id or 'CLARITY' in rule_id or 'WORD_CHOICE' in rule_id:
                error_type = 'STYLE_ERROR'

            penalty = penalty_map.get(error_type, 3) # Get penalty, default 3
            total_penalty += penalty
        
        # Calculate grammar score based on the number of errors
        if len(words) > 0:
            # Penalize more for denser errors
            scaled_penalty = (total_penalty / len(words)) * 50 # Adjust 50 as a sensitivity factor
            grammar_score = max(LOWER_BOUND, 100 - scaled_penalty)
            grammar_score = min(UPPER_BOUND, grammar_score / 10)  # Scale to 0-10
        else:
            grammar_score = 10  # No words, perfect score by default
    else:
        print("Language tool not initialized. Grammar score cannot be calculated.")
        grammar_score = 0
    
    return grammar_score
