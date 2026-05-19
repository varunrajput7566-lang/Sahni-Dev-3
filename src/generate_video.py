import os
import sys
import time
import json
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure CI/CD friendly logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load ephemeral .env created by GitHub Actions
load_dotenv()

# Configuration & Secrets
API_KEY = os.getenv("TOONFLOW_API_KEY")
WORKSPACE_ID = os.getenv("TOONFLOW_WORKSPACE_ID")
API_BASE_URL = "https://api.toonflow.com/v1" # Adjust to actual Toonflow endpoint

if not API_KEY:
    logger.error("CRITICAL: TOONFLOW_API_KEY is missing from environment.")
    sys.exit(1)

# Constants
# 23 minutes in seconds (Leaves 2 minutes for artifact upload and cleanup)
MAX_EXECUTION_TIME_SECONDS = 23 * 60  
START_TIME = time.time()

class APIError(Exception):
    """Custom exception for Toonflow API failures."""
    pass

def calculate_dynamic_duration(text: str) -> int:
    """
    Calculates video duration based on word count.
    Supports English, Hindi (Devanagari), and Hinglish natively.
    Average speaking rate: ~130 words per minute (approx 2.1 words per second).
    """
    # UTF-8 encoded text splitting handles Devanagari and Latin characters identically
    word_count = len(text.split())
    
    # 2.1 words per second + 3 seconds buffer for intro/outro transitions
    calculated_seconds = int((word_count / 2.1) + 3)
    
    # Enforce bounds: minimum 5 seconds, maximum 1200 seconds (20 mins)
    final_duration = max(5, min(calculated_seconds, 1200))
    logger.info(f"Duration calculation: {word_count} words -> {final_duration} seconds.")
    return final_duration

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, APIError))
)
def initialize_video_generation(prompt: str, duration: int) -> str:
    """
    Submits the prompt to Toonflow to begin the rendering job.
    Includes aggressive exponential backoff for network instability.
    """
    url = f"{API_BASE_URL}/render/start"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-Workspace-ID": WORKSPACE_ID
    }
    payload = {
        "script": prompt,
        "duration": duration,
        "resolution": "1080p",
        "language_auto_detect": True # Ensure API handles Hindi/Hinglish TTS natively
    }

    logger.info("Initiating Toonflow render job...")
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    
    if not response.ok:
        logger.error(f"API rejection: {response.status_code} - {response.text}")
        raise APIError(f"Failed to start job: {response.status_code}")
    
    job_id = response.json().get("job_id")
    logger.info(f"Job successfully queued. Job ID: {job_id}")
    return job_id

def poll_render_status(job_id: str) -> str:
    """
    Polls the API until the video is ready. 
    Implements a hard exit if approaching the GitHub Actions 25-min timeout limit.
    """
    url = f"{API_BASE_URL}/render/status/{job_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    # Dynamic polling interval to avoid rate limits
    poll_interval = 15 
    
    while True:
        elapsed_time = time.time() - START_TIME
        if elapsed_time > MAX_EXECUTION_TIME_SECONDS:
            logger.error("CRITICAL: Approaching GitHub Actions 25-minute limit. Gracefully aborting to prevent runner penalty.")
            sys.exit(1)

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            status = data.get("status")
            
            if status == "completed":
                download_url = data.get("download_url")
                logger.info("Render complete! Download URL acquired.")
                return download_url
            elif status in ["failed", "error"]:
                logger.error(f"Toonflow rendering failed: {data.get('error_message')}")
                sys.exit(1)
            else:
                logger.info(f"Status: {status.upper()}... Elapsed: {int(elapsed_time)}s. Waiting {poll_interval}s.")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Polling network error (ignoring and retrying): {e}")

        time.sleep(poll_interval)
        # Increase polling interval slightly over time up to 30 seconds
        poll_interval = min(30, poll_interval + 2)

def download_video(download_url: str):
    """Streams the video file to disk in chunks to minimize memory footprint."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / "final_render.mp4"
    
    logger.info(f"Downloading video to {file_path}...")
    try:
        with requests.get(download_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download video artifact: {e}")
        sys.exit(1)

def main():
    # 1. Capture Environment Inputs
    # Force UTF-8 decoding to safely handle Devanagari script via GitHub inputs
    raw_prompt = os.getenv("PROMPT", "").encode('utf-8').decode('utf-8')
    raw_duration = os.getenv("TARGET_DURATION", "0")

    if not raw_prompt:
        logger.error("No prompt provided. Exiting.")
        sys.exit(1)

    # 2. Determine Duration
    duration = int(raw_duration) if raw_duration.isdigit() else 0
    if duration == 0:
        duration = calculate_dynamic_duration(raw_prompt)

    # 3. Execute Pipeline
    try:
        job_id = initialize_video_generation(raw_prompt, duration)
        video_url = poll_render_status(job_id)
        download_video(video_url)
    except Exception as e:
        logger.error(f"Pipeline crashed gracefully: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
