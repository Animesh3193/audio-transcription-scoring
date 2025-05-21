from sentence_transformers import util
from sentence_transformers import SentenceTransformer

LOWER_BOUND = 0
UPPER_BOUND = 10

embedding_model = SentenceTransformer("avsolatorio/GIST-Embedding-v0", )


def calculate_relevancy_score(transcribe_output, topic):
    """
    Calculate relevancy score based on the transcribe output.
    """
    relevancy_score = 5.0

    # Fetch Text
    audio_text = transcribe_output[0].text
    clean_topic = topic.lower().strip()
    clean_audio_text = audio_text.lower().strip()

    # Encode the text
    topic_embedding = embedding_model.encode(clean_topic, convert_to_tensor=True)
    audio_embedding = embedding_model.encode(clean_audio_text, convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(topic_embedding, audio_embedding)

    # Calculate the average cosine similarity score
    avg_cosine_score = cosine_scores.mean().item()

    # Normalizing the score between 0 to 10
    relevancy_score = (avg_cosine_score + 1) / 2 * 10
    
    # Rounding of ensuring the score is between 0 to 10
    relevancy_score = round(max(LOWER_BOUND, min(relevancy_score, UPPER_BOUND)), 2)

    return relevancy_score