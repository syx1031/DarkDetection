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


def detect_outside_interface(client, video, start_time=None, end_time=None):
    class Outside_Interface(BaseModel):
        go_outside: bool = Field(..., description="Whether the user gets out of the app or not.")
        outside_interface_type: Literal["Home Screen", "Control Center", "Notification Center", "Browser", "App Switcher", "Others"] = Field(..., description="The interface type, should be in one of the following strings: 'Home Screen', 'Control Center', 'Notification Center', 'Browser', 'App Switcher', 'Others'.")
        go_outside_time: str = Field(..., description="The timestamp when the user gets out of the app, should be represented in the format 'mm:ss'.")
        resume_app_time: str = Field(..., description="The timestamp when the user returns to the app, should be represented in the format 'mm:ss'.")
        thinking: str = Field(..., description="Your detailed thinking process.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['go_outside_time'] = self.validate_timestamp(data['go_outside_time'])
            data['resume_app_time'] = self.validate_timestamp(data['resume_app_time'])
            super().__init__(**data)

    if end_time and not start_time:
        # prompt_detect_outside_interface = f'''
        #     Context:
        #     1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        #         a. The persistent red-bordered circle represents the current position of the cursor.
        #         b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        #         c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
        #
        #     Task:
        #     1. Check whether the user resumed the app from an interface out of the app within the two seconds before {end_time}. Note that you only need to examine events that occurred within the two seconds before {end_time}; events after it should be ignored.
        #     2. If your answer to Task 1 is yes, then identify the type ("Home Screen/Control Center/Notification Center/Browser/App Switcher/Other") of this outside interface, and identify when the user previously left the app and went to the external interface prior to this app resumption.
        # '''

        # prompt_detect_outside_interface = f'''
        #     Context:
        #     1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        #         a. The persistent red-bordered circle represents the current position of the cursor.
        #         b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        #         c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
        #     2. In the bottom-right corner of the video, the user has included two pieces of information: Frame ID and timestamp, which indicate the current frame’s position in the full frame sequence and its corresponding time in the video, respectively.
        #
        #     Task:
        #     1. Check whether the user resumed the app from an interface out of the app within the two seconds before {end_time}. Note that you only need to examine events that occurred within the three seconds before {end_time}; events after it should be ignored.
        #     2. If your answer to Task 1 is yes, you must clearly identify the exact frame where you observed the outside interface, report the frame id and timestamp in the bottom-right corner, then reanalyze this frame and reflect on whether your previous conclusion involved hallucination—for example, whether there was in fact no outside interface present in this frame.
        #     3. If you think there is indeed an outside interface, then identify the type ("Home Screen/Control Center/Notification Center/Browser/App Switcher/Other") of it.
        # '''

        prompt_detect_outside_interface = f'''
            Context:
            1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
                a. The persistent red-bordered circle represents the current position of the cursor.
                b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
                c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

            Task:
            1. Check whether the user resumed the app from an interface out of the app within the two seconds before {end_time}. Note that you only need to examine events that occurred within the three seconds before {end_time}; events after it should be ignored.
            2. If your answer to Task 1 is yes, you must clearly identify the exact frame where you observed the outside interface, then reanalyze this frame and reflect on whether your previous conclusion involved hallucination—for example, whether there was in fact no outside interface present in this frame.
            3. If you think there is indeed an outside interface, then identify the type ("Home Screen/Control Center/Notification Center/Browser/App Switcher/Other") of it.
        '''

        prompt_reasoning = '''
            Special Notice: 
            1. In the bottom-right corner of the video, the user has included two pieces of information: Frame ID and timestamp, which indicate the current frame’s position in the full frame sequence and its corresponding time in the video, respectively.
            2. I have reviewed this timestamp and confirmed that the user did not visit any external interface. In other words, for this sample, you should have output: {"go_outside": false, "outside_interface_type": "Others", "go_outside_time": "00:00", "resume_app_time": "00:00"}. However, in several previous queries, you consistently concluded that the user went to the Home Screen or Control Center at this timestamp. Please now provide your current judgment, and try to analyze the reasons behind the earlier incorrect assessments. You must clearly identify the exact frame where you observed the Home Screen or Control Center, and reports the frame id and timestamp in the bottom-right corner of this frame.
        '''

        prompt_rag_1 = f'''
            ###
            Exemplars:
                You must carefully distinguish between two situations: (1) the user leaving the app to visit an external interface such as "Home Screen", "Control Center", "Browser", and "App Switcher", and (2) brief black screens that occur before an ad finishes loading. Do not confuse the two situations. The following image is an example of a Home Screen, which belongs to the situation (1).
        '''
        image_shot_2 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\outside_interface\\6450103715-home-screen.jpg"
        prompt_rag_2 = open(image_shot_2, 'rb').read()
        prompt_rag_3 = '''The following image is an example of a Control Center, which belongs to the situation (1).'''
        image_shot_4 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\outside_interface\\1664705223-control-center.jpg"
        prompt_rag_4 = open(image_shot_4, 'rb').read()
        prompt_rag_5 = '''The following images show the black screen that appears before an ad finishes loading, which belongs to situation (2).'''
        image_shot_6 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\outside_interface\\6450103715-black-screen-1.jpg"
        prompt_rag_6 = open(image_shot_6, 'rb').read()
        image_shot_7 ="E:\\DarkDetection\\Gemini2.5Pro\\local_database\\outside_interface\\1479305181-black-screen-2.jpg"
        prompt_rag_7 = open(image_shot_7, 'rb').read()
        image_shot_8 = "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\outside_interface\\6453159988-black-screen-3.jpg"
        prompt_rag_8 = open(image_shot_8, 'rb').read()

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=Outside_Interface,
            topP=0.01,
            topK=1,
            temperature=0.0,
        )

        content = types.Content(
            parts=[
                types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type='video/mp4')),
                types.Part(text=prompt_detect_outside_interface),
                # types.Part(text=prompt_reasoning),
                types.Part(text=prompt_rag_1),
                types.Part.from_bytes(data=prompt_rag_2, mime_type='image/jpeg'),
                types.Part(text=prompt_rag_3),
                types.Part.from_bytes(data=prompt_rag_4, mime_type='image/jpeg'),
                types.Part(text=prompt_rag_5),
                types.Part.from_bytes(data=prompt_rag_6, mime_type='image/jpeg'),
                types.Part.from_bytes(data=prompt_rag_7, mime_type='image/jpeg'),
                types.Part.from_bytes(data=prompt_rag_8, mime_type='image/jpeg'),
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        print("Check Outside Interface:")
        print(response.text)
        return json.loads(response.text)


if __name__ == "__main__":
    client = get_client(key='AIzaSyDYMd2HNlDMZH7yJKDx16v-SDUralwcBEM')

    for cache in client.caches.list():
        client.caches.delete(cache.name)

    video_local_paths = [
        # "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\outside_interface\\6450103715-尚宇轩.mp4",
        "E:\\DarkDetection\\dataset\\syx\\us\\6450103715-尚宇轩.mp4",
        # "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\outside_interface\\1479305181-尚宇轩.mp4",
        "E:\\DarkDetection\\dataset\\syx\\us\\1479305181-尚宇轩.mp4",
        # "E:\\DarkDetection\\Gemini2.5Pro\\local_database\\outside_interface\\6453159988-尚宇轩.mp4",
        "E:\\DarkDetection\\dataset\\syx\\us\\6453159988-尚宇轩.mp4",
        # "E:\\DarkDetection\\dataset\\syx\\us\\1574455218-尚宇轩.mp4",
        # "E:\\DarkDetection\\dataset\\syx\\us\\6463493974-尚宇轩.mp4",
        # "E:\\DarkDetection\\dataset\\syx\\us\\6450351607-尚宇轩.mp4",
        # "E:\\DarkDetection\\dataset\\syx\\us\\6450103715-尚宇轩.mp4",
    ]

    end_times = [
        "04:05",
        "03:47",
        "01:02",
        "04:39",
        "03:08",
        "03:36",
        "04:06",
    ]


    # cloud_name = [
    #     # "files/c3ims4vz8bk8",
    #     "files/34bqprm3x9vw", # raw video
    #     # "files/wp71ravioym2",
    #     "files/hprs1tnlbwj8", # raw video
    #     # "files/4eocotpifdpr",
    #     "files/669qchxbx20o", # raw video
    #     "files/g4gmibqrlwpz",
    #     "files/tqz0tn8s8edh",
    #     "files/vk8tl22abtr4",
    #     "files/40ixknwougph",
    # ]

    cloud_name = [
        "files/8jg4clge0fwq", # raw video
        # "files/sb1ebeo40uxq",
        "files/6ekdupw25zpq", # raw video
        # "files/ezfu33l0mcg8",
        "files/r2tn17zmttwr", # raw video
        # "files/fbz602rp8wwt",
        "files/g4gmibqrlwpz",
        "files/tqz0tn8s8edh",
        "files/vk8tl22abtr4",
        "files/40ixknwougph",
    ]

    for i in range(len(video_local_paths)):
        video_local_path = video_local_paths[i]
        end_time = end_times[i]

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

        detect_outside_interface(client, video_file, end_time=end_time)
