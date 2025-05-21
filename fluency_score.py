import re
from collections import OrderedDict
import pandas as pd

LOWER_BOUND = 0
UPPER_BOUND = 10

def add_pauses_to_df(pause_df, minimum_pause_duration, punctuation):
    for index, row in pause_df.iterrows():
        if row['pause'] > minimum_pause_duration:
            pause_df.at[index, 'pause_valid'] = row['pause']
            
            if any(char in punctuation for char in index):
                pause_df.at[index, 'linguistic_pause'] = row['pause']
                pause_df.at[index, 'hesitation_pause'] = 0
            else:
                pause_df.at[index, 'hesitation_pause'] = row['pause']
                pause_df.at[index, 'linguistic_pause'] = 0
            
        else:
            pause_df.at[index, 'pause_valid'] = 0
            pause_df.at[index, 'hesitation_pause'] = 0
            pause_df.at[index, 'linguistic_pause'] = 0
    return pause_df


def calculate_fluency_score(transcribe_output):
    """
    Calculate fluency score based on the transcribe output.
    """
    
    fluency_score = 0
    # Fetch Text
    audio_text = transcribe_output[0].text
    
    word_timestamps = transcribe_output[0].timestamp['word']
    
    full_time_scale = word_timestamps[-1]['end'] - word_timestamps[0]['start']
    sentence_length = len(word_timestamps)

    pause_taken = OrderedDict()

    
    for index in range(sentence_length):
        
        word = word_timestamps[index]['word']
        if index == sentence_length - 1:
            time_taken = 0
        else:
            time_taken = word_timestamps[index+1]['start'] - word_timestamps[index]['end']
        
        word_duration = word_timestamps[index]['end'] - word_timestamps[index]['start']
        pause_taken[word] = {'pause': time_taken, 'duration': word_duration}
        
    pause_df = pd.DataFrame.from_dict(pause_taken, orient='index')


    minimum_pause_duration = 0.2

    punctuation = ".?!,;:-"

    pause_df = add_pauses_to_df(pause_df, minimum_pause_duration, punctuation)
    
    
    total_pause_time_sec = pause_df['pause_valid'].sum()
    num_total_pauses = len(pause_df[pause_df['pause_valid'] > 0])
    num_valid_pauses = len(pause_df[pause_df['linguistic_pause'] > 0])
    num_hesitation_pauses = len(pause_df[pause_df['hesitation_pause'] > 0])

    valid_pause_durations = pause_df['linguistic_pause'].sum()
    hesitation_pause_durations = pause_df['hesitation_pause'].sum()
    
    avg_pause_duration_sec = total_pause_time_sec / num_total_pauses if num_total_pauses > 0 else 0
    avg_valid_pause_duration_sec = valid_pause_durations / num_valid_pauses if num_valid_pauses > 0 else 0
    avg_hesitation_pause_duration_sec = hesitation_pause_durations / num_hesitation_pauses if num_hesitation_pauses > 0 else 0

    speaking_time_of_words = full_time_scale - total_pause_time_sec

    # Calculate rates
    speech_rate_wpm = (pause_df.shape[0] / full_time_scale) * 60 if full_time_scale > 0 else 0
    articulation_rate_wpm = (pause_df.shape[0] / speaking_time_of_words) * 60 if speaking_time_of_words > 0 else 0

    pauses_per_minute = (num_total_pauses / full_time_scale) * 60 if full_time_scale > 0 else 0
    hesitation_pauses_per_minute = (num_hesitation_pauses / full_time_scale) * 60 if full_time_scale > 0 else 0

    
    # Identify filled pauses
    filled_pauses = ["um", "uh", "ah", "err", "hmm", "like", "you know", "i mean", "so"]

    matching_filled_pauses = [re.findall(pause, audio_text) for pause in filled_pauses]
    num_filled_pauses = sum(len(matches) for matches in matching_filled_pauses)
    filled_pauses_percentage = (num_filled_pauses / pause_df.shape[0]) * 100 if pause_df.shape[0] > 0 else 0
    
    
    # --- Simple Fluency Score (Heuristic) ---
    fluency_score = 0

    # Speech Rate Component (target 120-150 WPM)
    if 110 <= speech_rate_wpm <= 160:
        fluency_score += 30 # Optimal range
    elif 90 <= speech_rate_wpm <= 180:
        fluency_score += 15 # Acceptable range
    else:
        fluency_score += 5 # Too slow/fast
    
    # Pause Component (fewer and shorter hesitation pauses are better)
    # Penalize for more hesitation pauses and longer average hesitation pauses
    if num_hesitation_pauses == 0:
        fluency_score += 20
    else:
        fluency_score += max(0, 20 - (hesitation_pauses_per_minute * 2.5) - (avg_hesitation_pause_duration_sec * 10))


    # Filled Pauses Component (fewer is better)
    fluency_score += max(0, 20 - (filled_pauses_percentage * 3))

    # Consistency (articulation rate vs. speech rate)
    if articulation_rate_wpm > speech_rate_wpm * 1.05 and articulation_rate_wpm < speech_rate_wpm * 1.5:
        fluency_score += 10 # Good balance, suggests efficient speaking between pauses
        
    # Total Fluency Score with normalization and bounds
    fluency_score = max(LOWER_BOUND, min(fluency_score/UPPER_BOUND, UPPER_BOUND))
    return fluency_score
