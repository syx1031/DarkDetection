import re
import json
from pydantic import BaseModel, Field
from google.genai import types
from typing import Literal

from utils import send_request


def detect_outside_interface(client, video, start_time=None, end_time=None):
    class Outside_Interface(BaseModel):
        go_outside: bool = Field(..., description="Whether the user gets out of the app or not.")
        outside_interface_type: Literal["Home Screen", "Control Center", "Notification Center", "Browser", "App Switcher", "Others"] = Field(..., description="The interface type, should be in one of the following strings: 'Home Screen', 'Control Center', 'Notification Center', 'Browser', 'App Switcher', 'Others'.")
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
        prompt_detect_outside_interface = f'''
            Context:
            1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
                a. The persistent red-bordered circle represents the current position of the cursor. 
                b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
                c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

            Task: 
            1. Check whether the user resumed the app from an interface out of the app within the two seconds before {end_time}. Note that you only need to examine events that occurred within the two seconds before {end_time}; events after it should be ignored.
            2. If your answer to Task 1 is yes, then identify the type ("Home Screen/Control Center/Notification Center/Browser/App Switcher/Other") of this outside interface, and identify when the user previously left the app and went to the external interface prior to this app resumption.
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