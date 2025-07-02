from google.genai import types
from google import genai
from pydantic import BaseModel, Field, confloat, validator
from typing import Literal
import re
import time
import json
from moviepy import VideoFileClip
import traceback

from utils import upload_file, send_request, get_client, dump_upload_files, time_to_seconds, seconds_to_mmss


class AdSegment(BaseModel):
    start_timestamp: str = Field(..., description="The timestamp at which the ad appear: should be represented in the format 'mm:ss'.")
    end_timestamp: str = Field(..., description="The timestamp at which the ad end: should be represented in the format 'mm:ss'.")
    full_screen: bool = Field(..., description="Whether the ad is a full-screen ad at the beginning of it.")
    description: str = Field(..., description="Briefly describe the ad’s position and content in one to two sentences.")
    thinking: str = Field(..., description="Explain in detail the reasoning behind your judgment that an advertisement appeared during this time period.")

    # 自定义验证器确保 period 格式合法
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        if not re.match(r"^\d{2}:\d{2}$", value):
            raise ValueError("timestamp must be in 'mm:ss' format.")
        return value

    def __init__(self, **data):
        # 手动调用验证器
        data['start_timestamp'] = self.validate_timestamp(data['start_timestamp'])
        data['end_timestamp'] = self.validate_timestamp(data['end_timestamp'])
        super().__init__(**data)


# 用于整体结构的 list 类型
AdSegmentList = list[AdSegment]


def detect_ads(client, video):
    prompt_detect_ads = '''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

    Task: 
    1. At what time did advertisements appear (use the format xx:xx–xx:xx)? List all time periods during which ads appeared. Note: (1) Although App Store pages or prompt pages requesting users to rate the app can be part of an ad, they are not considered ad interfaces when standalone. (2) The interface for paid ad removal is part of the app’s functional UI and does not count as an ad. (3) Interfaces that merely request the user to watch an ad (e.g., “Watch an ad for...”) are not considered ads.
    2. Reconsider the ad time intervals you identified in Task 1---especially those with similar content or consecutive time intervals---and determine whether they represent different stages of the same ad (e.g., a video followed by an playable demo) or different UI components of the same stage (e.g., a full-screen video and a banner with summary information below it). Then treat these intervals as a single ad for further analysis. List all time periods during which ads appeared again after your reconsideration.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdSegmentList,
        topP=0.01,
        topK=1,
        temperature=0.0,
    )
    contents = [video, prompt_detect_ads]

    # Send request with function declarations
    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        # contents=[video_file, prompt],
        contents=contents,
        config=config,
    )

    # print("Detect Ad:")
    # print(response.text)
    return json.loads(response.text)


def generate_content(contents):
    parts = []
    for content in contents:
        if "text" in content.keys():
            parts.append(types.Part(text=content["text"]))
        elif "img" in content.keys():
            parts.append(types.Part.from_bytes(data=content["img"], mime_type='image/jpeg'))
        elif "video" in content.keys():
            parts.append(types.Part(file_data=types.FileData(file_uri=content["video"], mime_type='video/mp4')))

    return parts


def recheck_ads(client, video, start_time, end_time):
    class AdAttribution(BaseModel):
        start_time: str = Field(...,
                                description="The timestamp when the ad starts, should be represented in the format 'mm:ss'. If no ad occurs, set this attribution to '00:00'.")
        end_time: str = Field(...,
                              description="The timestamp when the ad ends, should be represented in the format 'mm:ss'. If no ad occurs, set this attribution to '00:00'.")
        full_screen: bool = Field(..., description="Whether the ad is a full-screen ad at the beginning of it.")
        thinking: str = Field(..., description="Provide detailed analyze about above attributions of the ad.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['start_time'] = self.validate_timestamp(data['start_time'])
            data['end_time'] = self.validate_timestamp(data['end_time'])
            super().__init__(**data)

    # start_time_mmss = seconds_to_mmss(start_time)
    # end_time_mmss = seconds_to_mmss(end_time)
    prompt_recheck_ad = [{"text": f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

    Task: An ad may be displayed in the video during the period {start_time}-{end_time}. Please review this period and verify the following attributions of the ad:
    1. Check the few seconds before {start_time} to determine whether the preceding content also belongs to the same ad. Adjust the ad’s start time accordingly based on your analysis. A common scenario is that the app initially displays a non-full-screen ad interface (which contains ad content rather than merely prompting the user to watch an ad). Subsequently, due to user gestures or automatic transitions, the ad switches to a full-screen interface.
    2. Check the few seconds after {end_time} to determine whether the subsequent content also belongs to the same ad. Adjust the ad’s end time accordingly based on your analysis. 
    3. Check the beginning of the ad. If the ad occupies the entire screen or the majority, it should be classified as a "full-screen ad"; conversely, if the ad does not start in full-screen but later becomes full-screen, it should not be classified as a "full-screen ad".
    '''}]

    prompt_rag_1 = [{'text': f'''
        ###
        Exemplars: Typically, an ad may consist of multiple interfaces, such as a video, a playable game demo, and a static page summarizing product information in the end of the ad. To accurately identify an ad, you need to detect all interfaces that belong to it. 
        The following examples illustrate cases where you previously failed to recognize the complete sequence of ad interfaces. Please refer to these cases and determine whether the current ad shares the same interface display sequence.
        '''}]
    prompt_rag_2 = []
    image_shot_2_1 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\6463493974-a1-f1.jpg"
    prompt_rag_2.append({'img': open(image_shot_2_1, 'rb').read()})
    image_shot_2_2 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\6463493974-a1-f2.jpg"
    prompt_rag_2.append({'img': open(image_shot_2_2, 'rb').read()})
    prompt_rag_2.append({'text': '''The two images above show an ad composed of two interfaces. First, the ad presents a video. Then, it transitions to a static interface summarizing product information, including the product name “Dynasty Legends 2,” the price “Free,” a “GET” button, and the third-party ad platform “Unity Ads.” You previously failed to identify the second static interface and instead mistakenly treated the video as the complete ad.'''})

    prompt_rag_3 = []
    image_shot_3_1 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\6463493974-a2-f1.jpg"
    prompt_rag_3.append({'img': open(image_shot_3_1, 'rb').read()})
    image_shot_3_2 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\6463493974-a2-f2.jpg"
    prompt_rag_3.append({'img': open(image_shot_3_2, 'rb').read()})
    image_shot_3_3 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\6463493974-a2-f3.jpg"
    prompt_rag_3.append({'img': open(image_shot_3_3, 'rb').read()})
    image_shot_3_4 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\6463493974-a2-f4.jpg"
    prompt_rag_3.append({'img': open(image_shot_3_4, 'rb').read()})
    prompt_rag_3.append({'text': '''The four images above show an ad composed of four interfaces. First, the ad plays a video, and at the end of the video, it displays an interface as shown in image 1, featuring text such as “全服百万玩家” and “球星正版授权,” along with the Google Play and App Store logos. Next, the ad presents an interactive interface as shown in image 2, where a hand-shaped icon entices the user to tap on a central chest. After the user taps, the ad triggers a redirection to the app store, leaving the current app and attempting to show the app’s detail page. However, image 3 indicates that the app is not available in the user’s current region. Finally, when the user uses the app switcher to return to the original app, they see a final static interface shown in image 4, displaying the ad product name “全明星街球派对” and a “GET” button. You previously identified only the ad interfaces shown in images 1, 2, and 3, but failed to recognize that the interface the user saw after returning to the app was also part of the same ad.'''})

    prompt_rag_4 = []
    image_shot_4_1 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\6447191495-a3-f1.jpg"
    prompt_rag_4.append({'img': open(image_shot_4_1, 'rb').read()})
    image_shot_4_2 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\6447191495-a3-f2.jpg"
    prompt_rag_4.append({'img': open(image_shot_4_2, 'rb').read()})
    image_shot_4_3 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\6447191495-a3-f3.jpg"
    prompt_rag_4.append({'img': open(image_shot_4_3, 'rb').read()})
    prompt_rag_4.append({'text': '''The three images above depict an ad composed of three interfaces. It begins with a video as shown in image 1. After the video finishes, it transitions to a playable demo as shown in image 2. Finally, it displays the interface in image 3, which informs the user of the advertised app’s name, “Traffic Escape!”, and includes a “GET” button to encourage user interaction. You previously detected only the ad video and failed to identify the playable demo that appeared after the video ended.'''})

    prompt_rag_5 = []
    image_shot_5_1 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\1574455218-a4-f1.jpg"
    prompt_rag_5.append({'img': open(image_shot_5_1, 'rb').read()})
    image_shot_5_2 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\1574455218-a4-f2.jpg"
    prompt_rag_5.append({'img': open(image_shot_5_2, 'rb').read()})
    image_shot_5_3 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\1574455218-a4-f3.jpg"
    prompt_rag_5.append({'img': open(image_shot_5_3, 'rb').read()})
    prompt_rag_5.append({"text": '''The three images above show an ad consisting of three interfaces. Before the main ad content begins, Unity Ads platform uses the interfaces shown in images 1 and 2 to collect user information and obtain consent. Then, it displays the main ad content shown in image 3. You previously recognized only the main content in image 3, but it is important to note that the interfaces in images 1 and 2 are also part of the ad.'''})

    prompt_rag_6 = []
    image_shot_6_1 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\1607742297-a5-f1.jpg"
    prompt_rag_6.append({'img': open(image_shot_6_1, 'rb').read()})
    image_shot_6_2 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\483922001-a5-f2.jpg"
    prompt_rag_6.append({'img': open(image_shot_6_2, 'rb').read()})
    image_shot_6_3 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\ad\\1498061633-a5-f3.jpg"
    prompt_rag_6.append({'img': open(image_shot_6_3, 'rb').read()})
    prompt_rag_6.append({'text': '''
        The three images above show Google ad feedback pages in different sizes and positions. These interfaces only appear after the user closes the current ad, serving to inform the user and provide a feedback channel. Therefore, you should not consider this type of interface as part of the ad.
    '''})

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdAttribution,
        topP=0.01,
        topK=1,
        temperature=0.0,
    )

    # parts = generate_content([{"video": video_file.uri}] + prompt_recheck_ad + prompt_rag_1 + prompt_rag_2 + prompt_rag_3 + prompt_rag_4 + prompt_rag_5)
    # parts = generate_content([{"video": video_file.uri}] + prompt_recheck_ad)
    parts = generate_content([{"video": video_file.uri}] + prompt_recheck_ad + prompt_rag_6)

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=types.Content(
            parts=parts
        ),
        config=config,
    )

    # print("Recheck Ad:")
    # print(response.text)
    return json.loads(response.text)


def run_detect(client, video):
    ads_time = detect_ads(client, video)
    print("Ad_Detect:")
    print(json.dumps(ads_time, indent=4))

    for ad in ads_time:
        start_time = ad["start_timestamp"]
        end_time = ad["end_timestamp"]
        recheck_ads_time = recheck_ads(client, video, start_time, end_time)
        print(f"Recheck Ad {start_time}-{end_time}:")
        print(json.dumps(recheck_ads_time, indent=4))


if __name__ == "__main__":
    client = get_client(key='AIzaSyDYMd2HNlDMZH7yJKDx16v-SDUralwcBEM')

    for cache in client.caches.list():
        client.caches.delete(cache.name)

    video_local_paths = [
        # "E:\\DarkDetection\\dataset\\syx\\us\\6463493974-尚宇轩.mp4",
        # "E:\\DarkDetection\\dataset\\syx\\us\\6447191495-尚宇轩.mp4",
        # "E:\\DarkDetection\\dataset\\syx\\us\\1574455218-尚宇轩.mp4",
        "E:\\DarkDetection\\dataset\\syx\\us\\1607742297-尚宇轩.mp4",
        "E:\\DarkDetection\\dataset\\syx\\us\\483922001-尚宇轩.mp4",
        "E:\\DarkDetection\\dataset\\syx\\us\\1498061633-尚宇轩.mp4",
    ]

    cloud_name = [
        # "files/b85o7xx69ovz",
        # "files/49rx9nfujq5q",
        # "files/7tdb3hwgwpw0",
        "files/uzzzu6uza0bx",
        "files/t9ig2cv35ttd",
        "files/it6j4uixf2b6",
    ]

    for i in range(len(video_local_paths)):
        video_local_path = video_local_paths[i]

        if cloud_name[i] != "changed":
            video_file = client.files.get(name=cloud_name[i])
        else:
            video_file = client.files.upload(file=video_local_path)
            while not video_file.state or video_file.state.name != "ACTIVE":
                # print("Processing video...")
                # print("File state:", video_file.state)
                time.sleep(5)
                video_file = client.files.get(name=video_file.name)

        print(video_file.name)

        run_detect(client, video_file)
