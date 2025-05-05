import os
import requests
import subprocess
import random
import json
import time
import csv
from PIL import Image
from bs4 import BeautifulSoup
from io import BytesIO

# Config
# csv_path = "names_datasets/celebrities.csv"
# csv_path = "names_datasets/youtubers_df.csv"
csv_path = (
    "names_datasets/Artists.csv"  # Path to the CSV file containing celebrity names
)
output_dir = "celebrity_dataset"
video_duration = 5  # seconds

os.makedirs(output_dir, exist_ok=True)


def load_celebrities_from_csv(filepath, max_celebrities=10):
    celebrities = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = list(csv.DictReader(csvfile))

        # # Sort by popularity (convert to float)
        # sorted_rows = sorted(reader, key=lambda x: float(x["popularity"]), reverse=True)

        for row in reader[:max_celebrities]:
            name = row.get("name")
            if name:
                celebrities.append(name)

    return celebrities


def download_image(celebrity):
    headers = {"User-Agent": "Mozilla/5.0"}
    query = celebrity.replace(" ", "+")
    url = f"https://www.google.com/search?tbm=isch&q={query}"
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    images = soup.find_all("img")

    for img in images[1:]:
        img_url = img.get("src")
        if img_url:
            try:
                img_response = requests.get(img_url)
                img = Image.open(BytesIO(img_response.content))
                save_dir = os.path.join(output_dir, celebrity.replace(" ", "_"))
                os.makedirs(save_dir, exist_ok=True)
                img.save(os.path.join(save_dir, "photo.jpg"))
                print(f"Downloaded image for {celebrity}")
                return
            except Exception as e:
                print(f"Failed to download image for {celebrity}: {e}")
                continue
    print(f"No suitable image found for {celebrity}")


def get_video_info(video_id):
    try:
        result = subprocess.run(
            ["yt-dlp", "-j", f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error fetching video info: {e}")
        return None


def seconds_to_hms(seconds):
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"


def download_random_clip(video_id, celebrity, title):
    video_info = get_video_info(video_id)
    if not video_info:
        return

    duration = video_info.get("duration", 0)
    if duration <= video_duration + 10:
        print(f"Video too short for {celebrity}")
        return

    start = random.randint(5, duration - video_duration - 5)
    end = start + video_duration

    start_hms = seconds_to_hms(start)
    end_hms = seconds_to_hms(end)

    safe_name = celebrity.replace(" ", "_")
    save_dir = os.path.join(output_dir, safe_name)
    os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, "video.%(ext)s")
    metadata_path = os.path.join(save_dir, "metadata.json")

    print(f"Downloading clip for {celebrity} from {start_hms} to {end_hms}")

    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--quiet",
        "--no-warnings",
        "--download-sections",
        f"*{start_hms}-{end_hms}",
        "-o",
        output_path,
        "-f",
        "mp4",
    ]
    subprocess.run(cmd)

    metadata = {
        "celebrity": celebrity,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "video_title": title,
        "start_time": start_hms,
        "end_time": end_hms,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def download_video(celebrity):
    print(f"\nSearching compilation video for {celebrity}...")
    search_terms = [
        "interview",
    ]
    search_query = f"{celebrity} {random.choice(search_terms)}"

    result = subprocess.run(
        ["yt-dlp", f"ytsearch5:{search_query}", "--flat-playlist", "-j"],
        capture_output=True,
        text=True,
    )
    videos = result.stdout.strip().split("\n")
    random.shuffle(videos)

    for video_json in videos:
        try:
            data = json.loads(video_json)
            video_id = data.get("id")
            title = data.get("title")
            if video_id:
                download_random_clip(video_id, celebrity, title)
                return
        except Exception as e:
            print(f"Skipping video due to error: {e}")
    print(f"No video downloaded for {celebrity}")


def build_dataset():
    celebrities = load_celebrities_from_csv(csv_path, max_celebrities=10)
    for celeb in celebrities:
        try:
            download_video(celeb)
            time.sleep(1)
            download_image(celeb)
            time.sleep(2)
        except Exception as e:
            print(f"Error processing {celeb}: {e}")


if __name__ == "__main__":
    build_dataset()
