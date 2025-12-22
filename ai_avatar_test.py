import requests

# -----------------------------------------------------------
# Set your API KEY and SIMLI Avatar Generation API Endpoint
# -----------------------------------------------------------
API_KEY = "2kvvhj12f0rskm5lqdnpt"

# Replace this with the actual Simli avatar video generation endpoint
API_ENDPOINT = "https://api.simli.com/v1/avatar/generate"


# -----------------------------------------------------------
# PUT YOUR AUDIO URL HERE (hosted on AWS S3 or public URL)
# -----------------------------------------------------------
audio_url = "https://adarsh-demo-neurolearn.s3.ap-southeast-2.amazonaws.com/learning-tts/jesc110/learning-tts-jesc110-1764549201.wav"


# JSON payload your provider expects
payload = {
    "input_audio_url": audio_url,
    "avatar": "default"           # change this if Simli uses avatar IDs
}

# Request headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "*/*"
}

print("Sending request to Simli AI Avatar API...")

# POST request → Stream enabled for video download
response = requests.post(
    API_ENDPOINT,
    json=payload,
    headers=headers,
    stream=True
)

content_type = response.headers.get("content-type", "")

# -----------------------------------------------------------
# Case 1: API returns JSON with video_url
# -----------------------------------------------------------
if "application/json" in content_type:
    data = response.json()
    print("\nAPI JSON Response:")
    print(data)

    # If Simli returns a video link directly
    if "video_url" in data:
        print("\nDownloading video from:", data["video_url"])

        video_resp = requests.get(data["video_url"])
        filename = "simli_avatar_output.mp4"

        with open(filename, "wb") as f:
            f.write(video_resp.content)

        print("\n✔ Video saved as:", filename)
        exit()

    else:
        print("\nNo direct video returned. Response:")
        print(data)
        exit()

# -----------------------------------------------------------
# Case 2: API returns direct video file stream
# -----------------------------------------------------------
print("Direct video stream detected. Saving file...")

output_file = "simli_avatar_output.mp4"
with open(output_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

print("\n✔ Video saved as:", output_file)
