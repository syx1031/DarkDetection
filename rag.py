from google.genai import Client
from google.genai import types
import pandas as pd
import glob
import os
import cv2
import torch
import clip
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import io
import argparse
import re
import numpy as np
import ffmpeg

import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_google_vertexai import VertexAI , ChatVertexAI , VertexAIEmbeddings

from utils import time_to_seconds, get_client, send_request, upload_file, dump_upload_files, generate_part, seconds_to_mmss, get_embed


kf_database = "local_database\\ad\\keyframes"


def sanitize(filename: str) -> str:
    # Windows 文件名非法字符：<>:"/\|?*
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


class GeminiEmbeddings(Embeddings):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.model_id = "models/text-embedding-004"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (documents)."""
        embeddings = []
        for text in texts:
            response = get_embed(
                client=self.client,
                model=self.model_id,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                )
            )
            embeddings.append(response)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        response = get_embed(
            client=self.client,
            model=self.model_id,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            )
        )
        return response


def convert_frames_to_jpeg_bytes(frames):
    frame_bytes_list = []

    for frame in frames:
        # Convert BGR (OpenCV) to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Save to in-memory buffer in JPEG format
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        frame_bytes = buffer.getvalue()

        frame_bytes_list.append(frame_bytes)

    return frame_bytes_list


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


def detect_semantic_changes(embeddings, threshold=0.3):
    """Return indices of semantic change frames"""
    change_indices = [0]
    for i in range(1, len(embeddings)):
        sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        if sim < 1 - threshold:
            change_indices.append(i)
    return change_indices


def extract_key_frames(video, start_time, end_time):
    frame_interval = 0.5  # 每秒提一帧
    semantic_threshold = 0.3  # 控制关键帧的“语义变化”敏感度

    embeddings, frames, frame_indices = extract_clip_embeddings_from_segment(
        video, start_time, end_time, frame_interval
    )

    keyframe_indices = detect_semantic_changes(embeddings, threshold=semantic_threshold)

    keyframe_timestamp = []
    for kf_index in keyframe_indices:
        keyframe_timestamp.append(seconds_to_mmss(time_to_seconds(start_time) + int(frame_interval * kf_index)))

    key_frames = []
    for idx, i in enumerate(keyframe_indices):
        key_frames.append(frames[i])

    return key_frames, keyframe_timestamp


def generate_video_summarize(client, video, start_time, end_time):
    prompt_summarize_video = '''
        Context:
        1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
            a. The persistent red-bordered circle represents the current position of the cursor. 
            b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
            c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
        2. The provided video captures the entire process of a full-screen ad from beginning to end. 
            a. Typically, an ad may consist of multiple interfaces, such as a video, a playable game demo, a landing page, and a static page summarizing information of the ad. 
            b. During the ad display process, the ad may switch between different interfaces. These transitions can be triggered by user actions--for example, tapping a button on the current interface that leads to a landing page, or tapping a close button that causes another interface to appear in an attempt to retain the user. Interface transitions can also occur automatically, such as when a video countdown ends and the video finishes playing.
            
        Task: You are an assistant tasked with summarizing the ad for retrieval. Provide a brief summary of this ad by describing each interface in the order it appears. For each interface, mainly summarize its content and type (such as video, static interface, et al.) and explain the reason for the transition to the next interface. 
        Please note: These summaries will be embedded and used to retrieve the raw video in a RAG system, so give a concise summary of the video that is well optimized for retrieval.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
    )

    start_offset = str(time_to_seconds(start_time)) + 's'
    end_offset = str(time_to_seconds(end_time)) + 's'
    fps = 2

    content = types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri=video.uri, mime_type='video/mp4'),
                video_metadata=types.VideoMetadata(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    fps=fps,
                )
            ),
            types.Part(text=prompt_summarize_video)
        ]
    )

    # Send request with function declarations
    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return response.text


def upload_file_and_generate_video_summarize(video_local_path, todo):
    client = get_client(local_path=video_local_path)

    try:
        video_file = upload_file(client, video_local_path)
    except Exception as e:
        # raise
        return {'broken_file_upload': True, 'error_information': traceback.format_exc()}
    dump_upload_files()

    ad_result = []
    for ad in todo:
        start_time = ad["start_time"]
        end_time = ad["end_time"]
        full_screen = ad["full_screen"]
        if full_screen:
            try:
                ad_summarize = generate_video_summarize(client, video_file, start_time, end_time)
            except Exception as e:
                # raise
                return {'broken_client': True, 'error_information': traceback.format_exc()}
            ad_result.append({**ad, 'summarize': ad_summarize})

            key_frames, kf_timestamp = extract_key_frames(video_local_path, start_time, end_time)
            key_frames_path = os.path.join(kf_database, sanitize(
                f"{os.path.basename(video_local_path).split('-')[0]}-{start_time}-{end_time}"))
            os.makedirs(key_frames_path, exist_ok=True)
            for file_path in glob.glob(os.path.join(key_frames_path, "*")):
                os.remove(file_path)
            for i, kf_ts in enumerate(kf_timestamp):
                key_frames[i].save(os.path.join(key_frames_path, f"{i}-{sanitize(kf_ts)}.jpg"))

    return {'summarize': ad_result}


def get_ad_list():
    ad_list = {}

    gt_videos = "E:\\DarkDetection\\dataset\\syx\\us"
    gt_file = "E:\\DarkDetection\\dataset\\syx\\ground_truth_syx_us.xlsx"
    # 读取 Excel 文件（默认读取第一张表）
    gt_df = pd.read_excel(gt_file)

    # 遍历每一行（从第一行数据开始，不包括表头）
    # 获取所有唯一的序号（按顺序去重）
    unique_ids = gt_df['appid'].drop_duplicates().tolist()

    for uid in unique_ids[:40]:
        # 找出这个序号对应的所有行
        group = gt_df[gt_df['appid'] == uid]

        # 外层处理：第一行
        first_row = group.iloc[0]
        if first_row.get('能否测试') == '可测试' and glob.glob(os.path.join(gt_videos, f'{uid}*')) and len(group) > 1:
            video = glob.glob(os.path.join(gt_videos, f'{uid}*'))[0]
            ad_list[video] = []

            for _, row in group.iloc[1:].iterrows():
                full_screen = bool(row.get('1.3是否全屏广告'))
                raw_timestamp = row.get('2广告出现时间')
                startstamp, endstamp = raw_timestamp.split('-', 1)

                ad_list[video].append({'full_screen': full_screen, 'start_time': startstamp, 'end_time': endstamp})

    return ad_list


def dump_result_file(path, result_dict):
    with open(path, "w") as f:
        json.dump(result_dict, f, indent=4)


def load_database():
    def create_multi_vector_retriever(
            vectorstore, video_summaries, videos
    ):
        """
        Create retriever that indexes summaries, but returns raw videos or images
        """

        # Initialize the storage layer
        store = InMemoryStore()
        id_key = "doc_id"

        # Create the multi-vector retriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        # Helper function to add documents to the vectorstore and docstore
        def add_documents(retriever, doc_summaries, doc_contents):
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

        # Add video into the database
        if video_summaries:
            add_documents(retriever, video_summaries, videos)

        return retriever

    embedding_function = GeminiEmbeddings(client=get_client())

    # The vectorstore to use to index the summaries
    vectorstore = Chroma(
        collection_name="rag_full_screen_ads",
        embedding_function=embedding_function,
    )

    videos = []
    video_summaries = []

    video_summaries_file = "rag_done_list.json"

    ad_list = json.loads(open(video_summaries_file, "r").read())
    for video, ads in ad_list.items():
        for ad in ads["summarize"]:
            videos.append(json.dumps({'video': video, 'start_time': ad["start_time"], 'end_time': ad["end_time"], 'summarize': ad["summarize"]}))
            video_summaries.append(ad["summarize"])

    # Create retriever
    retriever_multi_vector = create_multi_vector_retriever(
        vectorstore,
        video_summaries,
        videos,
    )

    # Create RAG chain
    # chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector)

    return retriever_multi_vector


def generate_exemplars_parts(query_result):
    parts = [generate_part('''
        [Some Exemplars]
        Exemplars: Typically, an ad may consist of multiple interfaces, such as a video, a playable game demo, and a static page summarizing product information in the end of the ad. To accurately identify an ad, you need to detect all interfaces that belong to it. 
        The following exemplars present ads similar to the current one. You should learn from them the typical sequence of interfaces and the content shown in each. Then, use this knowledge to refine your previous conclusion by checking whether the previously identified ad time period missed any interfaces or included inaccurate descriptions.
    ''', "t")]

    for doc in query_result:
        content = json.loads(doc)
        video = content["video"]
        start_time = content["start_time"]
        end_time = content["end_time"]

        # key_frames = extract_key_frames(video, start_time, end_time)
        kf_path_exemplar = os.path.join(kf_database, sanitize(f"{os.path.basename(video).split('-')[0]}-{start_time}-{end_time}"))
        kf_file_paths = glob.glob(os.path.join(f"{kf_path_exemplar}", "*"))
        sorted_kf_files = sorted(
            kf_file_paths,
            key=lambda path: int(os.path.basename(path).split('-')[0])
        )
        for key_frame_path in sorted_kf_files:
            parts.append(generate_part(open(key_frame_path, "rb").read(), "i"))

        parts.append(generate_part(f'''
            The above {len(sorted_kf_files)} images are sampled keyframes from an ad in chronological order. Here is the description of this ad:
        ''' + content["summarize"], "t"))

    parts.append(generate_part("[End of Exemplars]", "t"))

    return parts


if __name__ == "__main__":
    # extract_key_frames("E:\\DarkDetection\\dataset\\syx\\us\\6447110104-尚宇轩.mp4", "00:33", "1:23")

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='None')
    args = parser.parse_args()

    done_list = {}

    if args.c != "None":
        with open(args.c, "r") as f:
            done_list = json.load(f)

    todo_list = get_ad_list()

    list_need_summary = dict(list(todo_list.items()))
    if args.c != "None":
        for video, info in todo_list.items():
            if video in done_list.keys():
                last_info = done_list[video]
                if "broken_client" not in last_info and "broken_file_upload" not in last_info and "crashed_future" not in last_info:
                    list_need_summary.pop(video)

    # list_need_summary = dict(list(todo_list.items()))
    # for video, info in todo_list.items():
    #     if video not in [
    #         "E:\\DarkDetection\\dataset\\syx\\us\\6447110104-尚宇轩.mp4",
    #         # "E:\\DarkDetection\\dataset\\syx\\us\\6447461923-尚宇轩.mp4"
    #     ]:
    #         list_need_summary.pop(video)

    RESULT_LOCK = threading.Lock()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(upload_file_and_generate_video_summarize, video, todo): video for video, todo in list_need_summary.items()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            video = futures[future]
            print(f'[End] Future on video {video} has done.')
            try:
                done_list_this_video = future.result()
            except Exception as e:
                # raise
                print(f"[Error] Future failed on video {video}: {e}")
                traceback.print_exc()
                with RESULT_LOCK:
                    done_list[video] = {"crashed_future": True, "error_info": traceback.format_exc()}
                    dump_result_file("rag_done_list.json", done_list)
                continue

            with RESULT_LOCK:
                if "broken_client" in done_list_this_video.keys() or "broken_file_upload" in done_list_this_video.keys():
                    print(f"Gemini cannot respond with this api on video {video}")
                    done_list[video] = done_list_this_video
                    dump_result_file("rag_done_list.json", done_list)
                    continue

                done_list[video] = done_list_this_video
                dump_result_file("rag_done_list.json", done_list)

    # retriever = load_database()
    # query = "This video showcases a full-screen mobile ad for a Dominoes game.\n\n1.  **Initial Interface (03:53 - 03:54):** A black screen is displayed with an iOS timer in the top left, indicating a loading or waiting phase. This transitions automatically to reveal the ad banner.\n2.  **App Store Banner (03:54 - 03:55):** An App Store banner for \"Domino - Dominoes onlin...\" appears at the bottom of the black screen. This automatically leads to the main interactive ad content.\n3.  **Playable Dominoes Game (03:55 - 04:01):** The ad transitions to an interactive, playable demo of a multiplayer Dominoes game. The user actively participates by dragging and dropping domino tiles onto the board. The game progresses with turns taken by all players.\n4.  **Score Screen (04:01 - 04:02):** Upon completion of the game, a \"SCORE\" screen appears, displaying the points for each player. This automatically transitions to the game mode selection screen.\n5.  **Game Mode Selection (04:02 - 04:04):** An \"ONLINE GAME\" menu is shown, presenting options for different Dominoes game types such as \"DRAW GAME\" and \"ALL FIVES.\" The user then initiates closing the ad by swiping up the home indicator.\n6.  **App Minimized/Closed (04:04 - 04:05):** The ad (app) window minimizes to the iOS home screen view, and the user then swipes it away to fully close the application."
    # answer = retriever.invoke(query)
    # pass
