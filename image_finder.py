import os
import requests
import json
import boto3
from groq import Groq
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from urllib.parse import urlparse
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()

# --- CONFIGURATION ---
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/images"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "neurolearn"
COLLECTION_NAME = "files"

# AWS Config
AWS_AUDIO_BUCKET = os.getenv("AWS_AUDIO_BUCKET") or os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION") or "us-east-1"
AWS_IMAGE_PREFIX = "learning-images"

# Rate Limit: 4 calls per 1 second for Serper
CALLS = 4
PERIOD = 1

def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def get_groq_client():
    api_key_str = os.getenv("GROQ_API_KEY", "")
    api_keys = [k.strip() for k in api_key_str.split(',') if k.strip()]
    
    if not api_keys:
        raise ValueError("GROQ_API_KEY is not set")
        
    return Groq(api_key=api_keys[0])

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def get_educational_images(query):
    """
    Fetches images from Serper.dev with a strict rate limit.
    """
    payload = json.dumps({
        "q": f"{query} educational diagram",
        "num": 4,
        "autocorrect": True,
        "safe": "active"
    })
    
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(SERPER_URL, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        images = [img['imageUrl'] for img in results.get('images', [])]
        return images
    except Exception as e:
        print(f"Error fetching images for '{query}': {e}")
        return []

def generate_keywords_from_summary(summary_text):
    """
    Uses Groq to extract search keywords from the summary.
    """
    client = get_groq_client()
    
    prompt = f"""
    Analyze the following educational summary and extract 4 distinct, visual search keywords 
    that would yield good educational diagrams or illustrations.
    Return ONLY a JSON array of strings. No other text.
    
    Summary:
    {summary_text[:2000]}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts visual keywords."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        content = chat_completion.choices[0].message.content
        # Clean up code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()
        keywords = json.loads(content)
        return keywords if isinstance(keywords, list) else []
    except Exception as e:
        print(f"Error generating keywords: {e}")
        return []

def upload_image_to_s3(image_url, folder="default", target_filename=None):
    """
    Downloads image from URL, converts to PNG, and uploads to S3.
    Returns the public S3 URL.
    """
    if not AWS_AUDIO_BUCKET:
        print("AWS_AUDIO_BUCKET not set, skipping upload")
        return image_url

    try:
        # Download image
        resp = requests.get(image_url, stream=True, timeout=10)
        resp.raise_for_status()
        
        # Determine filename
        if target_filename:
            s3_key = f"{AWS_IMAGE_PREFIX}/{folder}/{target_filename}"
        else:
            # Fallback
            parsed = urlparse(image_url)
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = f"image_{int(os.urandom(4).hex(), 16)}.png"
            safe_name = secure_filename(filename)
            s3_key = f"{AWS_IMAGE_PREFIX}/{folder}/{safe_name}"

        # Convert to PNG using Pillow
        image_data = BytesIO(resp.content)
        img = Image.open(image_data)
        
        # Convert to RGB if necessary (e.g. for RGBA or P modes if saving as JPEG, but PNG handles RGBA)
        # Just ensuring it is loaded
        
        out_img = BytesIO()
        img.save(out_img, format='PNG')
        out_img.seek(0)
        
        # Upload
        s3 = boto3.client('s3', region_name=AWS_REGION)
        s3.upload_fileobj(
            out_img,
            AWS_AUDIO_BUCKET,
            s3_key,
            ExtraArgs={'ContentType': 'image/png'}
        )
        
        # Construct URL
        if AWS_REGION == 'us-east-1':
            return f"https://{AWS_AUDIO_BUCKET}.s3.amazonaws.com/{s3_key}"
        return f"https://{AWS_AUDIO_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        
    except Exception as e:
        print(f"Failed to upload {image_url} to S3: {e}")
        return image_url # Fallback to original URL. NOTE: Frontend might fail if it strictly expects PNG from bucket

def process_images_for_file(file_path, summary_text):
    """
    Full pipeline: Summary -> Keywords -> Images -> S3 -> MongoDB
    """
    if not summary_text:
        print("No summary text provided.")
        return []
        
    print(f"\n🖼️ Starting Image Finder for {file_path}...")
    
    # 1. Generate Keywords
    print("Step 1: Generating keywords...")
    keywords = generate_keywords_from_summary(summary_text)
    print(f"🔑 Keywords: {keywords}")
    
    if not keywords:
        print("No keywords generated.")
        return []

    all_images = []
    
    # 2. Find Images
    print("Step 2: Finding images...")
    for kw in keywords:
        imgs = get_educational_images(kw)
        if imgs:
            # Take the top 1 valid image for each keyword
            all_images.append(imgs[0])
            
    # Deduplicate
    unique_images = list(dict.fromkeys(all_images))
    print(f"🔍 Found {len(unique_images)} images.")
    
    # 3. Upload to S3
    print("Step 3: Uploading to S3...")
    final_urls = []
    
    # Clean up base file name for the folder and the image prefix
    # file_path might be "uploads\jesc110.pdf"
    base_filename_full = os.path.basename(file_path) # jesc110.pdf
    base_name_clean = os.path.splitext(base_filename_full)[0] # jesc110
    safe_folder = secure_filename(base_name_clean) # jesc110
    
    # For the individual image files, user wants: "jesc 110 learning-image-1"
    # I will use safe names: "jesc110-learning-image-1.png"
    
    for i, img_url in enumerate(unique_images):
        target_name = f"{safe_folder}-learning-image-{i+1}.png"
        
        s3_url = upload_image_to_s3(img_url, folder=safe_folder, target_filename=target_name)
        final_urls.append(s3_url)
        print(f"✅ Image Ready: {s3_url}")
        
    # 4. Update MongoDB
    print("Step 4: Updating MongoDB...")
    try:
        col = get_mongo_collection()
        res = col.update_one(
            {"filePath": file_path},
            {"$set": {"images": final_urls}}
        )
        
        if res.modified_count > 0:
            print(f"✅ Successfully saved {len(final_urls)} images to MongoDB for {file_path}")
        else:
            res = col.update_one(
                {"filename": base_filename_full},
                {"$set": {"images": final_urls}}
            )
            if res.modified_count > 0:
                 print(f"✅ Successfully saved {len(final_urls)} images to MongoDB for {base_filename_full} (by filename)")
            else:
                 print(f"ℹ️ Document matched but not modified (images might be same) or not found.")
                 
    except Exception as e:
        print(f"Error updating MongoDB: {e}")
            
    return final_urls

if __name__ == "__main__":
    # Test run
    test_summary = "Photosynthesis is the process used by plants to convert light energy into chemical energy."
    print("Testing pipeline with dummy summary...")
    # Mocking file path
    process_images_for_file("test_photosynthesis.pdf", test_summary)