# Use the official Python image
FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords

# Cloud Run requires app to listen on port 8080
ENV PORT 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
 
# https://hate-speech-app-1095002057086.us-central1.run.app
