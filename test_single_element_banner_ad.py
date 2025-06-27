import ffmpeg
from PIL import Image
import io
import torch
import clip
import numpy as np
import tqdm
import os
import glob
import json


def extract_frame_ffmpeg(video_path, timestamp_sec):
    out, _ = (
        ffmpeg
        .input(video_path, ss=timestamp_sec)
        .output('pipe:', vframes=1, format='image2pipe', vcodec='png')
        .run(capture_stdout=True, capture_stderr=True)
    )
    return Image.open(io.BytesIO(out))


def sliding_windows_from_bottom(img: Image.Image):
    width, height = img.size
    window_height = int(height / 6)
    stride = int(height / 12)
    max_top = int(height * 2 / 5)

    windows = []

    top = height - window_height  # 从底部开始
    while top >= max_top:
        box = (0, top, width, top + window_height)
        window = img.crop(box)
        windows.append(window)
        top -= stride

    return windows


def classify_video_frames_with_clip(video_path, frame_interval=3.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 分类标签描述
    class_names = [
        "An advertisement with a width close to or equal to the image width is located inside a rectangular frame.",
        "An ad featuring a 'GET' button",
        "An ad featuring a 'Play Now' button",
        "An ad featuring a 'Learn More' text",
        "An ad with an 'X' close button and a icon 'i' in the corner.",
        "An ad with an 'X' close button and a Google icon.",
    ]
    text_tokens = torch.cat([clip.tokenize(t) for t in class_names]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    examples_path = [
        "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\banner ad\\1607742297-a1.jpg",
        "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\banner ad\\6450351607-a2.jpg",
        "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\banner ad\\6450351607-a3.jpg",
        "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\banner ad\\1607742297-a4.jpg",
        "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\banner ad\\6453159988-a5.jpg",
        "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\banner ad\\6453159988-a6.jpg",
        "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\banner ad\\6453159988-a7.jpg",
        "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\banner ad\\09.jpg",
    ]
    image_examples = [Image.open(open(example, "rb")) for example in examples_path]
    reference_features = []
    for ref_img in image_examples:
        ref_input = preprocess(ref_img).unsqueeze(0).to(device)
        with torch.no_grad():
            ref_feat = model.encode_image(ref_input)
            ref_feat /= ref_feat.norm(dim=-1, keepdim=True)
            reference_features.append(ref_feat)
    reference_tensor = torch.cat(reference_features, dim=0)  # shape: [N, D]

    # 获取视频时长
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe["format"]["duration"])
    except Exception as e:
        print(f"[ERROR] Could not determine video duration: {e}")
        return []

    timestamps = np.arange(0, duration, frame_interval)

    results = {}

    basename = os.path.basename(video_path)
    os.makedirs(f"result_dir\\{basename}", exist_ok=True)
    for any_image in glob.glob(f"result_dir\\{basename}\\*"):
        os.remove(any_image)

    for t in tqdm.tqdm(timestamps, desc="Processing video"):
        try:
            image = extract_frame_ffmpeg(video_path, t)
            image.save(f"result_dir\\{basename}\\{t}.jpg", format="JPEG")

            os.makedirs(f"result_dir\\{basename}\\{t}", exist_ok=True)
            for any_image in glob.glob(f"result_dir\\{basename}\\{t}\\*"):
                os.remove(any_image)

            scores = []
            result_windows = []
            windows = sliding_windows_from_bottom(image)

            for i, window in enumerate(windows):
                window.save(f"result_dir\\{basename}\\{t}\\{i}.jpg")
                image_input = preprocess(window).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_feature = model.encode_image(image_input)
                    image_feature /= image_feature.norm(dim=-1, keepdim=True)

                with torch.no_grad():
                    text_sims = (image_feature @ text_features.T).squeeze(0)  # shape: [N_text]
                    text_max_sim = text_sims.max().item()
                    text_max_sim = (text_max_sim + 1) / 2  # 映射到 [0, 1]

                with torch.no_grad():
                    image_sims = (image_feature @ reference_tensor.T).squeeze(0)  # shape: [N]
                    image_max_sim = image_sims.max().item()
                    image_max_sim = (image_max_sim + 1) / 2

                final_score = 0.7 * image_max_sim + 0.3 * text_max_sim
                scores.append(final_score)

                result_windows.append({
                    "timestamp": t,
                    "image_max_sim": image_max_sim,
                    "text_max_sim": text_max_sim,
                    "final_score": final_score,
                })

            final_score = max(scores)
            results[t] = {"part": result_windows, "max_score": final_score}
        except Exception as e:
            print(f"[WARN] Failed at {t:.2f}s: {e}")
            continue

    return results


# 示例调用
if __name__ == "__main__":
    results = {}
    video_paths = glob.glob("E:\\DarkDetection\\dataset\\syx\\us\\*")
    for video_path in video_paths[:3]:
        result_this_video = classify_video_frames_with_clip(video_path, frame_interval=3.0)
        results[video_path] = result_this_video

    with open("result_dir\\banner_detect.json", "w") as f:
        json.dump(results, f, indent=4)
