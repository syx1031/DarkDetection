from google.genai import types
from types import SimpleNamespace
from google import genai
import time
import random
import threading
import os
import json


def time_to_seconds(time_str):
    """将 hh:mm:ss 或 mm:ss 转为秒数"""
    parts = list(map(int, time_str.strip().split(":")))
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    elif len(parts) == 3 and parts[0] == 0 and parts[1] <= 10 and parts[2] <= 59:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    elif len(parts) == 3 and parts[0] <= 10 and parts[1] <= 59 and parts[2] <= 999:
        m, s, ms = parts
        return m * 60 + s
    else:
        raise ValueError("时间格式错误，必须是 mm:ss, hh:mm:ss或者mm:ss:msms")


def seconds_to_mmss(seconds):
    """将秒数转为 mm:ss 格式"""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes):02d}:{int(seconds):02d}"


def generate_part(content, content_type):
    if content_type == "t":
        return types.Part(text=content)
    elif content_type == "i":
        return types.Part.from_bytes(data=content, mime_type='image/jpeg')
    elif content_type == "v":
        return types.Part(file_data=types.FileData(file_uri=content, mime_type='video/mp4'))


FREE_KEYS = [
    "AIzaSyBWdjtqfv8QMBpqaI7smQAArpL_I5dJAsU",
    "AIzaSyAr4kAGPP3V9MEjbFddzBU1JO6WZ7xNVGw",
    "AIzaSyAfL3fHwxU0m_azmGxD6EERN1hn-s2_BOI",
    "AIzaSyDaYTs-CA65NcLt8T840Xhm8107xq3XRqM",
    "AIzaSyDbZy4YSTor8_ej3j3lAtJ7-nDMURAHafE",
    "AIzaSyAb8rKNNGEIxWO9zFCKKoKaxp7gHnuIFXQ",
    "AIzaSyD7_zs0cjXvfa2BMNYlm-_Cyz0-cSmdc18",
    "AIzaSyCWXJtKakyNvSETWZyq_0KKk2LbxL7S5yw",
    "AIzaSyAWUS0RPKsdFTAwsKJFrV1AiiCB5ML3SQ8",
    "AIzaSyB4w4chjyM8Pn81DmC-eLjMWgw3zttb8sY",
    "AIzaSyD22lWTZq6SPGgeQicUYic8G1oukNnntxo",
    "AIzaSyA-bxStNkKeadvr4bsd3cd1wBJCpCTrrzA",
    "AIzaSyB_K--N-6ZmALvHHy1CAoSAGzEK-ZOhGJo",
    "AIzaSyB6mAvrTP7RkT5I7_Q4obsOqMh4e8PdwVQ",
    "AIzaSyDRAssxVtCecUgKS4yKzpqvdxUS6hhvTj0",
    "AIzaSyAtF3FGFiYj0QNNQ2elUiHcVOQJ0kJEpWU",
    "AIzaSyCFfoGAC4_ll41uWki90MF2AKtkP2AUx1E",
    "AIzaSyCnps8AgUcg7xy8JFCJcAYHaNUU82UXAeU",
    "AIzaSyDpTUAjosqMrXZo2jbOyCeOehw_4OCbf6I",
    "AIzaSyCBbmC5ScZODwgF9gpr_egmJ1K_pdYC3Uc",
    "AIzaSyDYMd2HNlDMZH7yJKDx16v-SDUralwcBEM",
    "AIzaSyCIk3IzgIvY1olSbWQc-v_cOGHZkTDm2GM",
    "AIzaSyBDWZ12GJEBelO8S63lp4YeVzTye92XX1Y",
    "AIzaSyC8Jdhw8IST_zdcmm95C7fMYYi-XpTeqzA",
]
PAID_KEYS = [
    "AIzaSyDNYdjTbR2MYpnkOE6I2L8SrkBK4be2ndg"
]

# 按照推荐使用的顺序从上到下排列
MODEL_LIST = [
    "models/gemini-2.5-flash",
    "gemini-2.5-pro",
    "models/gemini-2.5-flash-lite-preview-06-17",
    # "models/gemini-2.0-flash", 不支持思考
]

# MODEL_CODE = {
#     "gemini-2.5-pro": "gemini-2.5-pro",
#     "gemini-2.5-flash": "models/gemini-2.5-flash",
#     "gemini-2.5-flash-lite-preview-06-17": "models/gemini-2.5-flash-lite-preview-06-17",
#     "gemini-2.0-flash": "models/gemini-2.0-flash"
# }

# Configure the client and tools
FREE_API_KEYS = {key: {'client': genai.Client(api_key=key, http_options={'timeout': 600000}), 'available_model': {model: True for model in MODEL_LIST}, 'status': True, 'uploads': {}, 'timer': None, 'lock': threading.Lock()} for key in FREE_KEYS}
PAID_API_KEYS = {key: {'client': genai.Client(api_key=key, http_options={'timeout': 600000}), 'available_model': {model: True for model in MODEL_LIST}, 'status': True, 'uploads': {}, 'timer': None, 'lock': threading.Lock()} for key in PAID_KEYS}
ALL_API_KEYS = FREE_API_KEYS | PAID_API_KEYS

STATUS_LOCK = threading.Lock()

for key in ALL_API_KEYS.keys():
    upload_videos_file = f"UploadVideos/{key}.json"
    if os.path.exists(upload_videos_file):
        with open(upload_videos_file, "r") as f:
            ALL_API_KEYS[key]['uploads'] = json.load(f)


def get_client(key=None, local_path=None):
    with STATUS_LOCK:
        # print(f"{local_path} get client")
        if key:
            return ALL_API_KEYS[key]['client']

        # if local_path:
        #     for key in ALL_API_KEYS.keys():
        #         upload_videos = ALL_API_KEYS[key]['uploads']
        #         if local_path in upload_videos.keys() and ALL_API_KEYS[key]['status']:
        #             try:
        #                 cloud_name = upload_videos[local_path]["cloud_name"]
        #                 cloud_file = ALL_API_KEYS[key]['client'].files.get(name=cloud_name)
        #                 if cloud_file.state and cloud_file.state.name == "ACTIVE":
        #                     return ALL_API_KEYS[key]['client']
        #             except Exception as e:
        #                 pass

        """获取一个可用的 API key，如果免费都不可用，则返回付费 key"""
        available_free_apis = [api['client'] for api in FREE_API_KEYS.values() if api['status']]
        available_paid_apis = [api['client'] for api in PAID_API_KEYS.values()]

        if available_free_apis:
            return random.choice(available_free_apis)
        else:
            return random.choice(available_paid_apis)


def get_key_from_client(client):
    for key, value in ALL_API_KEYS.items():
        if client is value['client']:
            return key


def upload_file(client, local_path):
    key = get_key_from_client(client)

    with STATUS_LOCK:
        # print(f"{local_path} test cloud")
        upload_videos = ALL_API_KEYS[key]['uploads']

        if local_path in upload_videos.keys():
            cloud_name = upload_videos[local_path]["cloud_name"]
            try:
                cloud_file = client.files.get(name=cloud_name)
                if cloud_file.state and cloud_file.state.name == "ACTIVE":
                    video_file = cloud_file
                    return video_file
                else:
                    upload_videos.pop(local_path)
            except Exception as e:
                upload_videos.pop(local_path)

    try_count = 0
    while True:
        try:
            video_file = client.files.upload(file=local_path)
            break
        except Exception as e:
            try_count += 1
            if try_count >= 3:
                raise
            print(f'Upload failed: {get_key_from_client(client)}, try again after several minutes.')
            time.sleep(120)

    with STATUS_LOCK:
        # print(f"{local_path} update upload files")
        upload_videos = ALL_API_KEYS[key]['uploads']

        upload_videos[local_path] = {"cloud_name": video_file.name}
        upload_videos_file = f"UploadVideos/{key}.json"
        with open(upload_videos_file, "w") as f:
            json.dump(upload_videos, f, indent=4)

    # Poll until the video file is completely processed (state becomes ACTIVE).
    try_count = 0
    while not video_file.state or video_file.state.name != "ACTIVE":
        # print("Processing video...")
        # print("File state:", video_file.state)
        time.sleep(5)
        try:
            video_file = client.files.get(name=video_file.name)
        except Exception as e:
            try_count += 1
            if try_count >= 10:
                raise

    return video_file


def restore_key(key):
    with STATUS_LOCK:
        FREE_API_KEYS[key]['status'] = True
        print(f"[{time.strftime('%H:%M:%S')}] API {key} is now restored.")


def feedback(key, model):
    with STATUS_LOCK:
        if key in FREE_KEYS:
            FREE_API_KEYS[key]["available_model"][model] = False

            if all(model_status is False for model_status in FREE_API_KEYS[key]["available_model"].values()):
                FREE_API_KEYS[key]["status"] = False
                print(f"[{time.strftime('%H:%M:%S')}] API {key} marked as over quota.")

            # # 如果已有恢复定时器在运行就不重复设置
            # if FREE_API_KEYS[key]['timer'] and FREE_API_KEYS[key]['timer'].is_alive():
            #     return
            #
            # timer = threading.Timer(90000, restore_key, args=[key])
            # FREE_API_KEYS[key][timer] = timer
            # timer.start()
        else:
            print('Paid api may be over quota.')


def out_of_quota(key, model):
    with STATUS_LOCK:
        if key in FREE_KEYS:
            return not FREE_API_KEYS[key]["available_model"][model]
        else:
            return False


def get_available_model(key):
    with STATUS_LOCK:
        if key in FREE_KEYS:
            for model in MODEL_LIST:
                if FREE_API_KEYS[key]["available_model"][model]:
                    return model
            return None
        else:
            return MODEL_LIST[0]


def send_request(client, model, contents, config, retry_sec=120):
    if model not in MODEL_LIST:
        model = MODEL_LIST[0]

    if out_of_quota(get_key_from_client(client), model):
        model = get_available_model(get_key_from_client(client))

    try_counter = 0
    while True:
        try:
            # Send request with function declarations
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            # if not response or not response.text:
            #     print(f"Response is None from {get_key_from_client(client)}.")
            #     response = SimpleNamespace(text="[]")
            if response and response.text:
                break
        except Exception as e:
            print(f'Request failed: {get_key_from_client(client)}, try again after several minutes. Detail: {e}')

            if hasattr(e, 'code') and e.code == 500:
                pass
            elif hasattr(e, 'code') and e.code == 429:
                try:
                    if e.details['error']['details'][0]['violations'][0]['quotaId'] == "GenerateRequestsPerDayPerProjectPerModel-FreeTier":
                        feedback(get_key_from_client(client), model)
                        model = get_available_model(client)
                        if not model:
                            raise
                        try_counter = 0
                    # elif e.details['error']['details'][0]['violations'][0]['quotaId'] == "GenerateContentInputTokensPerModelPerMinute-FreeTier":
                except Exception as e_again:
                    pass

            try_counter += 1
            if try_counter >= 10:
                # feedback(get_key_from_client(client))
                raise
            time.sleep(retry_sec)

    return response


def get_embed(client, model, contents, config):
    try_counter = 0
    while True:
        try:
            # Send request with function declarations
            response = client.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )
            break
        except Exception as e:
            print(f'Embed failed: {get_key_from_client(client)}, try again after several minutes. Detail: {e}')
            try_counter += 1
            if try_counter >= 10:
                raise
            time.sleep(120)

    return response.embeddings[0].values


def dump_upload_files():
    with STATUS_LOCK:
        # print(f"dump files")
        for key in ALL_API_KEYS.keys():
            upload_videos_file = f"UploadVideos/{key}.json"
            with open(upload_videos_file, "w") as f:
                json.dump(ALL_API_KEYS[key]['uploads'], f, indent=4)
