import ffmpeg
from PIL import Image
import io
import torch
from utils import time_to_seconds
import numpy as np
import tqdm

import clip


def extract_frame_ffmpeg(video_path, timestamp_sec):
    # 调用 ffmpeg-python 提取图像数据并读取到内存中
    out, _ = (
        ffmpeg
        .input(video_path, ss=timestamp_sec)
        .output('pipe:', vframes=1, format='image2pipe', vcodec='png')
        .run(capture_stdout=True, capture_stderr=True)
    )
    return Image.open(io.BytesIO(out))


def extract_clip_embeddings_from_segment(video_path, start_time, end_time, frame_interval=1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    start_sec = time_to_seconds(start_time)
    end_sec = time_to_seconds(end_time)

    timestamps = np.arange(start_sec, end_sec, frame_interval)

    embeddings = []
    frames = []
    frame_times = []

    for t in tqdm(timestamps, desc="Extracting frames via ffmpeg"):
        try:
            img = extract_frame_ffmpeg(video_path, t)
            # img.save(f"temp/{t}.jpg", format="JPEG")
            image_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image_tensor)
                embedding = embedding.cpu().numpy()[0]
                embeddings.append(embedding)
                frames.append(img)
                frame_times.append(t)
        except Exception as e:
            print(f"[WARN] Failed at {t:.2f}s: {e}")
            continue

    return embeddings, frames, frame_times
