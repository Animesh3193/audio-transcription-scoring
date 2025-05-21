import os
import uuid
from typing import Dict, Any
import asyncio
import aiofiles
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import UploadFile, File, BackgroundTasks, HTTPException, status, APIRouter
from fastapi.responses import JSONResponse

from transcribe import transcribe_audio_with_nemo_parakeet
from fluency_score import calculate_fluency_score
from vocabulary_score import calculate_vocabulary_score
from grammar_score import calculate_grammar_score
from relevancy_score import calculate_relevancy_score

router = APIRouter(prefix="/audio", tags=["transcribe"])

# --- In-memory storage for results ---
# In a real application, use a database (e.g., Redis, PostgreSQL, MongoDB)
results_db: Dict[str, Dict[str, Any]] = {}
processing_status: Dict[str, str] = {} # To track if a task is 'processing' or 'completed'


# --- Configuration ---
UPLOAD_DIRECTORY = "uploaded_audio"
# Create the directory if it doesn't exist
Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)

# --- Background Task Execution ---
executor = ThreadPoolExecutor(max_workers=4) # Use a ThreadPoolExecutor for CPU-bound tasks

async def run_scoring_in_background(unique_id: str, transcribed_output: Any, topic: str):
    """
    Runs all scoring functions in parallel using ThreadPoolExecutor.
    """
    processing_status[unique_id] = "processing"
    results_db[unique_id] = {"status": "processing", "transcription": transcribed_output, "topic": topic}

    loop = asyncio.get_event_loop()
    
    # Run CPU-bound tasks in the executor
    fluency_task = loop.run_in_executor(executor, calculate_fluency_score, transcribed_output)
    vocabulary_task = loop.run_in_executor(executor, calculate_vocabulary_score, transcribed_output)
    grammar_task = loop.run_in_executor(executor, calculate_grammar_score, transcribed_output)
    relevancy_task = loop.run_in_executor(executor, calculate_relevancy_score, transcribed_output, topic)

    # Await all tasks to complete
    fluency_result, vocabulary_result, grammar_result, relevancy_result = await asyncio.gather(
        fluency_task, vocabulary_task, grammar_task, relevancy_task
    )

    # Store results
    results_db[unique_id]["fluency"] = round(fluency_result, 2)
    results_db[unique_id]["vocabulary"] = round(vocabulary_result, 2)
    results_db[unique_id]["grammar"] = round(grammar_result, 2)
    results_db[unique_id]["relevancy"] = round(relevancy_result, 2)
    results_db[unique_id]["status"] = "completed"
    processing_status[unique_id] = "completed"


@router.post("/input", summary="Submit audio for transcription and analysis",
          description="Upload an audio file and provide a topic for analysis. "
                      "The audio will be transcribed, and then fluency, vocabulary, "
                      "grammar, and topic relevancy will be calculated in the background.")
async def process_input(
    audio_file: UploadFile = File(..., description="Audio file to transcribe (e.g., WAV, MP3)"),
    topic: str = File(..., description="The topic related to the audio content")
):
    """
    Handles audio file upload and topic submission for background processing.
    """
    # Generate a unique ID for this request
    unique_id = str(uuid.uuid4())
    
    # --- SAVE THE UPLOADED AUDIO FILE ---
    # Construct a unique filename, preserving the original extension
    file_extension = os.path.splitext(audio_file.filename)[1]
    saved_filename = f"{unique_id}{file_extension}"
    file_path = Path(UPLOAD_DIRECTORY) / saved_filename

    try:
        async with aiofiles.open(file_path, 'wb') as f:
            while contents := await audio_file.read(1024 * 1024): # Read in 1MB chunks
                await f.write(contents)
        print(f"Audio file saved to: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Could not save audio file: {e}")
    # Simulate Nemo Parakeet transcription
    transcribed_output = await transcribe_audio_with_nemo_parakeet(str(file_path))

    # Store initial status and transcription
    results_db[unique_id] = {"status": "transcribing", "transcription": transcribed_output[0].text, "topic": topic}
    processing_status[unique_id] = "transcribing"
    print(f"Transcription completed for {unique_id}: {transcribed_output[0].text}")
    # Add the intensive scoring task to background
    # This will run `run_scoring_in_background` without blocking the API response
    background_tasks = BackgroundTasks()
    background_tasks.add_task(run_scoring_in_background, unique_id, transcribed_output, topic)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "message": "Audio submitted for processing. Results will be available shortly.",
            "unique_id": unique_id,
            "transcription": transcribed_output[0].text # Optionally return transcription immediately
        },
        background=background_tasks
    )

@router.get("/results/{unique_id}", summary="Retrieve analysis results",
         description="Get the analysis results (fluency, vocabulary, grammar, relevancy) "
                     "for a given unique ID. Returns 'processing' if not yet complete.")
async def get_results(unique_id: str):
    """
    Retrieves the analysis results for a given unique ID.
    """
    if unique_id not in results_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unique ID not found.")

    transcribe_status = processing_status.get(unique_id, "unknown")
    
    if transcribe_status == "completed":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "scores":{
                "fluency": results_db[unique_id].get("fluency"),
                "vocabulary": results_db[unique_id].get("vocabulary"),
                "grammar": results_db[unique_id].get("grammar"),
                "relevancy": results_db[unique_id].get("relevancy")
            },
            "message": "Analysis completed.",
            "unique_id": unique_id,
            }
            )
    else:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED, # Still processing
            content={
                "status": transcribe_status,
                "message": "Processing in progress. Please try again later.",
                "unique_id": unique_id,
                "transcription": results_db[unique_id].get("transcription") # Show transcription if available
            }
        )

@router.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Audio Analysis API. Go to /docs for Swagger UI."}
