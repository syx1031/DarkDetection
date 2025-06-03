from google.genai import types
from google import genai
import time
import random
import threading


# Configure the client and tools
FREE_API_KEYS = [
genai.Client(api_key="AIzaSyBWdjtqfv8QMBpqaI7smQAArpL_I5dJAsU"),
genai.Client(api_key="AIzaSyAr4kAGPP3V9MEjbFddzBU1JO6WZ7xNVGw"),
genai.Client(api_key="AIzaSyAfL3fHwxU0m_azmGxD6EERN1hn-s2_BOI"),
genai.Client(api_key="AIzaSyDaYTs-CA65NcLt8T840Xhm8107xq3XRqM"),
genai.Client(api_key="AIzaSyDbZy4YSTor8_ej3j3lAtJ7-nDMURAHafE"),
genai.Client(api_key="AIzaSyAb8rKNNGEIxWO9zFCKKoKaxp7gHnuIFXQ"),
genai.Client(api_key="AIzaSyD7_zs0cjXvfa2BMNYlm-_Cyz0-cSmdc18"),
genai.Client(api_key="AIzaSyCWXJtKakyNvSETWZyq_0KKk2LbxL7S5yw"),
genai.Client(api_key="AIzaSyAWUS0RPKsdFTAwsKJFrV1AiiCB5ML3SQ8"),
genai.Client(api_key="AIzaSyB4w4chjyM8Pn81DmC-eLjMWgw3zttb8sY"),
genai.Client(api_key="AIzaSyD22lWTZq6SPGgeQicUYic8G1oukNnntxo"),
]

PAID_API_KEY = [
genai.Client(api_key="AIzaSyDNYdjTbR2MYpnkOE6I2L8SrkBK4be2ndg")
]

# 状态缓存
api_status = {key: True for key in FREE_API_KEYS}

# 锁用于保护对 api_status 的并发访问
status_lock = threading.Lock()

# 恢复定时器管理（防止多个定时器重复设定）
recovery_timers = {}


def get_client():
    """获取一个可用的 API key，如果免费都不可用，则返回付费 key"""
    available_free_keys = [key for key in FREE_API_KEYS if api_status.get(key, True)]

    if available_free_keys:
        return random.choice(available_free_keys)
    else:
        return PAID_API_KEY


def feedback(api_key):
    """标记某个 key 为不可用，并设置 2 分钟后恢复"""
    if api_key in FREE_API_KEYS:
        with status_lock:
            api_status[api_key] = False
            print(f"[{time.strftime('%H:%M:%S')}] API {api_key} marked as over quota.")

        # 如果已有恢复定时器在运行就不重复设置
        if api_key in recovery_timers and recovery_timers[api_key].is_alive():
            return

        def restore_key():
            with status_lock:
                api_status[api_key] = True
                print(f"[{time.strftime('%H:%M:%S')}] API {api_key} is now restored.")

        timer = threading.Timer(120, restore_key)
        recovery_timers[api_key] = timer
        timer.start()
    else:
        print('Paid api may be over quota.')


def time_to_seconds(time_str):
    """将 hh:mm:ss 或 mm:ss 转为秒数"""
    parts = list(map(int, time_str.strip().split(":")))
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    elif len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    else:
        raise ValueError("时间格式错误，必须是 mm:ss 或 hh:mm:ss")


def seconds_to_mmss(seconds):
    """将秒数转为 mm:ss 格式"""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes):02d}:{int(seconds):02d}"


def send_request(model, contents, config):
    while True:
        client_now = get_client()
        try:
            # Send request with function declarations
            response = client_now.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            break
        except Exception as e:
            print('Request failed, try to use another key.')
            feedback(client_now)

    return response
