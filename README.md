# audio-transcription-scoring
Opensource transcribing audio and analyzing fluency, vocabulary, grammar, and topic relevancy.


This project implements a FastAPI application that transcribes spoken audio, analyzes its linguistic aspects (fluency, vocabulary, grammar), and assesses its relevancy to a given topic. The core processing is handled in the background, allowing for quick API responses.

## Features

* **Audio Transcription:** Transcribes uploaded audio files into text.
* **Background Processing:** Performs linguistic analysis and topic relevancy checks asynchronously.
* **Unique Request IDs:** Provides a unique ID for each submission to retrieve results later.
* **Swagger UI:** Automatically generated API documentation for easy interaction and testing.
* **Dockerized Deployment:** Optimized for Docker, pre-downloading models and data during build.

## Core Logic & Calculations

The application processes transcribed audio and a given topic using several linguistic metrics:

1.  ### Fluency Score
    * **Logic:** Currently, this is a simplified estimation based on the total number of words in the transcribed text and a simulated speaking time. In a real-world scenario, precise fluency requires word-level timestamps from the ASR model to calculate metrics like Words Per Minute (WPM), pause rate, and speaking rate variability.
    * **Score:** A score between 1 and 10 is assigned, favoring typical conversational speaking rates.

2.  ### Vocabulary Score
    * **Logic:** Evaluates the diversity and richness of the vocabulary used.
        * **MTLD (Measure of Textual Lexical Diversity):** Calculates the average segment length where a certain type-token ratio is maintained. Higher MTLD indicates more sustained lexical diversity.
        * **HD-D (Hypergeometric Distribution Diversity):** A statistical measure modeling the probability of encountering unique words in random samples, providing a robust, length-independent diversity score.
        * **Average Word Length:** Simple measure of word complexity.
        * **Word Commonness/Rarity:** Assesses how common or rare the words used are, based on the Brown Corpus frequency distribution.
        * **Readability Metrics (Flesch Reading Ease, Gunning Fog Index):** Provide an indication of how easy or difficult the text is to understand.
    * **Tools:** `lexicalrichness` for MTLD and HD-D, `nltk` for word processing and commonness, `textstat` for readability.
    * **Score:** A composite score between 1 and 10 derived from the various vocabulary metrics.

3.  ### Grammar Score
    * **Logic:** Detects and quantifies grammatical errors in the transcribed text. It attempts to filter out common spoken disfluencies before analysis to focus on structural correctness. Errors are categorized and penalized based on a severity system (e.g., subject-verb agreement errors incur higher penalties than simple punctuation issues).
    * **Tool:** `language_tool_python` (an interface to LanguageTool).
    * **Score:** A score between 1 and 10, where a higher score indicates fewer or less severe grammatical errors.

4.  ### Relevancy Score
    * **Logic:** Measures the semantic similarity between the user-provided topic and the transcribed audio content. This is achieved by converting both the topic and the audio transcript into dense numerical vectors (embeddings) that capture their meaning. Cosine similarity is then calculated between these vectors.
    * **Embedding Model:** Uses the **`avsolatorio/GIST-Embedding-v0`** model. This is a lightweight model known for its performance (ranked 31st on the MTEB Leaderboard as of May 2025) and efficiency, making it suitable for practical applications.
    * **Link to Model:** [https://huggingface.co/avsolatorio/GIST-Embedding-v0](https://huggingface.co/avsolatorio/GIST-Embedding-v0)
    * **Score:** A score between 1 and 10, representing how semantically similar the response is to the topic (1 being irrelevant, 10 being highly relevant).

## Audio Transcription (Nemo Parakeet)

While the current implementation uses a mock transcriber for demonstration, a real deployment would integrate with a powerful Automatic Speech Recognition (ASR) model like **NVIDIA NeMo's Parakeet-TDT-0.6B-v2**.

* **Model:** `nvidia/parakeet-tdt-0.6b-v2`
* **Link to Model:** [https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
* **Note:** This is a very recent upgrade in the transcription world and NVIDIA parakeet model occurs to be one of the best model in Open-ASR leader board in May-2025, being 600-million-parameter automatic speech recognition model its faster to run and provide good results.

## How to Build and Run (Docker)

This application is designed for Docker deployment to ensure all dependencies, including large models and NLTK data, are managed efficiently.

1.  **Prerequisites:**
    * Docker installed on your system.
    * A good internet connection for the initial Docker build (model and data downloads).

2.  **Project Setup:**
    * Place all your Python files (`main.py`, `download_models.py`, `download_nltk_data.py`, `requirements.txt`) and the `Dockerfile` in the same directory.

3.  **Prepare `requirements.txt`:**
    ```
    fastapi
    uvicorn[standard]
    python-multipart
    sentence-transformers
    scipy
    scikit-learn
    nltk
    lexicalrichness
    textstat
    language-tool-python
    aiofiles
    nemo_toolkit[asr]
    ```
    * **Crucially:** If you're building for GPU (recommended for ASR), ensure your `Dockerfile` explicitly installs a CUDA-compatible PyTorch version *before* `nemo_toolkit[asr]`.

4.  **Create `download_models.py`:**
    The models come pre-loaded and downloaded when building the Docker, which contain the Transcription model and the embedding model
    ```python
    # download_models.py
    import nemo.collections.asr as nemo_asr
    from sentence_transformers import SentenceTransformer
    import os

    os.environ['HF_HOME'] = "/root/.cache/huggingface"

    print("Downloading Sentence-BERT model: avsolatorio/GIST-Embedding-v0")
    embed_model = SentenceTransformer('avsolatorio/GIST-Embedding-v0')
    print("Embedding Model downloaded successfully!")
    print("Downloading Transcription ASR Model: nvidia/parakeet-tdt-0.6b-v2")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    print("Transcription Model downloaded successfully!")
    ```

5.  **Create `download_nltk_data.py`:**
    ```python
    # download_nltk_data.py
    import nltk
    import os

    nltk_data_dir = os.environ.get('NLTK_DATA', '/usr/local/nltk_data')
    nltk.data.path.append(nltk_data_dir)

    print(f"Downloading NLTK data to: {nltk_data_dir}")

    def download_if_not_found(package_name):
        try:
            nltk.data.find(package_name)
            print(f"  {package_name} already found.")
        except nltk.downloader.DownloadError:
            print(f"  Downloading {package_name}...")
            nltk.download(package_name, download_dir=nltk_data_dir)
            print(f"  {package_name} downloaded.")

    download_if_not_found('corpora/stopwords')
    download_if_not_found('tokenizers/punkt')
    download_if_not_found('corpora/wordnet')
    download_if_not_found('corpora/brown')

    print("NLTK data download script finished.")
    ```

6.  **Create `Dockerfile`:** (Refer to the previous detailed `Dockerfile` provided, including multi-stage build, `HF_HOME`, `NLTK_DATA`, and the `download_models.py` and `download_nltk_data.py` steps.)

7.  **Build the Docker Image:**
    Navigate to your project directory in the terminal and run:
    ```bash
    docker build -t audio-analysis-app .
    ```
    This step will download all necessary Python packages, the GIST embedding model, and NLTK data *during the build process*. This might take some time on the first run.

8.  **Run the Docker Container:**
    ```bash
    docker run -p 5000:5000 audio-analysis-app
    ```
    This will start the FastAPI application inside a Docker container, accessible on your host machine at port 8000.

9.  **Access the API:**
    Open your web browser and go to `http://127.0.0.1:5000/docs` to interact with the API using Swagger UI.

## API Endpoints

* **`POST /input`**:
    * **Description:** Upload an audio file and provide a text topic. The audio is transcribed (mocked), and analysis tasks are initiated in the background.
    * **Returns:** A `unique_id` to track the processing status and results.
* **`GET /results/{unique_id}`**:
    * **Description:** Retrieve the analysis results for a given `unique_id`.
    * **Returns:** The analysis results (fluency, vocabulary, grammar, relevancy scores) if processing is `completed`, or a `202 Accepted` status if still `processing`.

## Local Development (Without Docker for NLTK/Model Downloads)

If you prefer to run locally for development and handle NLTK/model downloads manually:

1.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download NLTK Data:**
    Run the following in your Python environment once:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('brown')
    ```
3.  **Download GIST Embedding Model:**
    The `SentenceTransformer` will download the model automatically when it's first initialized in `main.py` (e.g., `embedding_model = SentenceTransformer('avsolatorio/GIST-Embedding-v0')`). This happens once.
4.  **Run Application:**
    ```bash
    uvicorn main:app --reload
    ```