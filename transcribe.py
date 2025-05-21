import nemo.collections.asr as nemo_asr


async def transcribe_audio_with_nemo_parakeet(audio_file):
    """
    Transcribe audio using Nemo Parakeet.
    """
    # Load the model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    
    # Transcribe the audio file
    transcription = asr_model.transcribe([audio_file], timestamps=True)

    return transcription
