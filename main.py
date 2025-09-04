from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import whisper
from transformers import pipeline
import tempfile
import os
from typing import Dict, Any
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app instance
# Think of this as setting up your restaurant
app = FastAPI(
    title="Audio Summarizer API",
    description="Convert MP3 files to text and generate summaries",
    version="1.0.0"
)

# CORS Middleware - this allows your React app to talk to this API
# Without this, browsers block requests between different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Create React App dev server  
        "https://*.vercel.app",   # Your Vercel deployment
        "https://*.ngrok.io",     # Ngrok tunnels
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model = None
summarizer = None

@app.on_event("startup")
async def load_models():
    """
    This function runs when the server starts up.
    It's like preheating your oven and preparing your ingredients.
    """
    global whisper_model, summarizer
    
    logger.info("Loading AI models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("tiny", device="cpu")
        logger.info("Whisper model loaded successfully")
        
        logger.info("Loading summarization model...")
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1 
        )
        logger.info("Summarization model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.get("/")
async def root():
    """
    Root endpoint - like the front door of your restaurant.
    Just returns a welcome message.
    """
    return {"message": "Audio Summarizer API is running!"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint - tells you if the models are ready.
    Like checking if your chefs are ready to cook.
    """
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "summarizer_loaded": summarizer is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio to text.
    This endpoint takes an MP3 file and returns the text transcript.
    
    Parameters:
    - file: The uploaded MP3 file
    
    Returns:
    - transcript: The text content of the audio
    - duration: How long the audio was
    """
    
    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400, 
            detail="File must be an audio file"
        )
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing audio file: {file.filename}")
        
        result = whisper_model.transcribe(temp_file_path)
        
        os.unlink(temp_file_path)
        
        transcript = result["text"].strip()
        language = result.get("language", "unknown")
        
        logger.info("Transcription completed successfully")
        
        return {
            "transcript": transcript,
            "language": language,
            "filename": file.filename
        }
        
    except Exception as e:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/summarize")
async def summarize_text(data: Dict[str, str]):
    """
    Summarize text content.
    
    Parameters:
    - data: Dictionary with "text" key containing the text to summarize
    
    Returns:
    - summary: The summarized text
    - original_length: Number of words in original text
    - summary_length: Number of words in summary
    """
    
    text = data.get("text", "").strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Text too short to summarize")
    
    try:
        logger.info("Starting text summarization...")
        
        max_length = min(1000, len(text.split()) // 4)
        min_length = max(50, max_length // 4)
        
        summary_result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        summary = summary_result[0]['summary_text']
        
        logger.info("Summarization completed successfully")
        
        return {
            "summary": summary,
            "original_length": len(text.split()),
            "summary_length": len(summary.split()),
            "compression_ratio": round(len(summary.split()) / len(text.split()), 2)
        }
        
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    """
    Complete pipeline: Upload MP3 → Transcribe → Summarize
    This is the main endpoint that does everything in one go.
    """
    
    try:
        logger.info(f"Starting complete processing for: {file.filename}")
        
        await file.seek(0)
        
        transcribe_result = await transcribe_audio(file)
        transcript = transcribe_result["transcript"]
        
        summary_result = await summarize_text({"text": transcript})
        
        logger.info("Complete processing finished successfully")
        
        return {
            "filename": file.filename,
            "transcript": transcript,
            "summary": summary_result["summary"],
            "language": transcribe_result["language"],
            "stats": {
                "original_words": summary_result["original_length"],
                "summary_words": summary_result["summary_length"],
                "compression_ratio": summary_result["compression_ratio"]
            }
        }
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
