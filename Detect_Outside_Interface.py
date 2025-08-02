import re
import json
from pydantic import BaseModel, Field
from google.genai import types
from typing import Literal

from utils import send_request, time_to_seconds, seconds_to_mmss


def detect_outside_interface(client, video, start_time=None, end_time=None):
    class Outside_Interface(BaseModel):
        go_outside: bool = Field(..., description="Whether the user gets out of the app or not.")
        # outside_interface_type: Literal["Home Screen", "Control Center", "Notification Center", "Browser", "App Switcher", "Others"] = Field(..., description="The interface type, should be in one of the following strings: 'Home Screen', 'Control Center', 'Notification Center', 'Browser', 'App Switcher', 'Others'.")
        outside_interface_type: Literal["Home Screen", "Control Center"] = Field(...,
            description="The interface type, should be in one of the following strings: 'Home Screen', 'Control Center', 'Notification Center'.")
        go_outside_time: str = Field(..., description="The timestamp when the user gets out of the app, should be represented in the format 'mm:ss'.")
        resume_app_time: str = Field(..., description="The timestamp when the user returns to the app, should be represented in the format 'mm:ss'.")

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

        prompt_detect_outside_interface = f'''
        Context: 
            1. You will be analyzing a video segment based on the following information:
                a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
                b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
                c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
                d. Frame id and timestamp: The bottom-right corner of the video displays the ID and timestamp of the current frame.
            2. Definition of "Outside Interface": Any screen, view, or overlay that DOES NOT belong to the app being used. This includes:
                - Home Screen: Characterized by a grid of app icons, a dock at the bottom, and a wallpaper background.
                - Control Center: An overlay (usually swiped from the top-right) with system control toggles (e.g., Wi-Fi, Brightness) over a blurred background.
                - Notification Center: An overlay (usually swiped from the top-center) displaying a list of notifications over a blurred background.
        
        Task: You must perform the following steps based on the video content leading up to {end_time}.
            Step 1: Check for App Resumption
                Analyze the video segment from {seconds_to_mmss(time_to_seconds(end_time) - 3)} to {end_time}. Then determine if an "App Resumption" occurs within this period. An "App Resumption" is defined as the moment the Target App's UI becomes the primary, full-screen view, immediately replacing a visible "Outside Interface".
            Step 2: Recheck your conclusion
                If your answer to Step 1 is yes, you must clearly identify the exact frame where you observed the outside interface (you need to provide the frame id and timestamp shown in the video), then reanalyze this frame and reflect on whether your previous conclusion involved hallucination—for example, whether there was only a black screen without outside interface present in this frame.
            Step 3: Provide Detailed Analysis (Only if Step 2 is "Yes")
            If a App Resumption event occurred, you must identify the following:
                2a. Interface Type Identification: Based on the visual evidence immediately preceding the resumption, identify the type of the "Outside Interface". Your answer must be one of the predefined categories: Home Screen, Control Center, Notification Center.
                2b. Departure Timestamp Identification: Scan the video prior to the resumption event to find the exact timestamp of the "departure event". A "departure event" is the moment the user initially left the Target App, causing its UI to be replaced by that specific "Outside Interface".
                
        Important: You should only identify Outside Interfaces that disappear between {seconds_to_mmss(time_to_seconds(end_time) - 3)} and {end_time}. Any interface disappearing outside this time frame should be ignored.
        
        Output: For each "Outside Interface" you find, provide the following details.
            - 'outside_interface_type': The type of Outside Interface ("Home Screen/Control Center/Notification Center/Browser/App Switcher/Other").
            - 'go_outside_time': The timestamp "mm:ss" when the Outside Interface begin to clearly appear.
            - 'resume_app_time': The timestamp "mm:ss" when the user returns to the app from the Outside Interface.
        '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=Outside_Interface,
        )

        content = [video, prompt_detect_outside_interface]

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Check Outside Interface:")
        # print(response.text)
        return json.loads(response.text)