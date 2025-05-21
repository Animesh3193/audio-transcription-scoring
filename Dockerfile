FROM python:3.10-slim

# Install dependencies for audio processing and ffmpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


# Set environment variables for non-interactive NLTK downloads
ENV NLTK_DATA=/usr/local/nltk_data

# --- Hugging Face Model Download Step ---
# Create a directory for Hugging Face cache and models
# This directory will be included in the final image
ENV HF_HOME="/root/.cache/huggingface"
RUN mkdir -p ${HF_HOME}/transformers \
           ${HF_HOME}/datasets \
           ${HF_HOME}/models \
    && chmod -R 777 ${HF_HOME} # Ensure permissions are adequate for the user later


WORKDIR /app


# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# This forces the model to be downloaded and cached within the build layer.
COPY download_parakeet_nltk.py .
RUN python download_parakeet_nltk.py

# Copy application code
COPY . .

# Expose port for API
EXPOSE 5000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
