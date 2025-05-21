FROM python:3.10-slim

# Set environment variables early
ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV NLTK_DATA=/usr/local/nltk_data
ENV HF_HOME="/root/.cache/huggingface"

# Install dependencies, including OpenJDK 17
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Create Hugging Face cache directories with appropriate permissions
RUN mkdir -p ${HF_HOME}/transformers \
           ${HF_HOME}/datasets \
           ${HF_HOME}/models \
    && chmod -R 777 ${HF_HOME}

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install HF KET
RUN pip install hf-xet

# Download models and NLTK data
COPY download_parakeet_nltk.py .
RUN python download_parakeet_nltk.py

# Copy the rest of the application code
COPY . .

# Expose the app port
EXPOSE 5000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
