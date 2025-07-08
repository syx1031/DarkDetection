import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Bbox import Location


def detect_shake_element_time_location(client, video, start_time=None, end_time=None):
    class ShakeElement(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which the element (text, icon, et al.) suggesting the user shake their phones appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(...,
                                   description="The location where the element suggesting the user shake their phones appears.")
        thinking: str = Field(...,
                              description="Explain in detail the reasoning behind your judgment.")

        # 自定义验证器确保 period 格式合法
        @classmethod
        def validate_timestamp(cls, value: str) -> str:
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp must be in 'mm:ss' format.")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            # data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    ShakeElementList = list[ShakeElement]

    prompt_detect_shake_element = f'''
        Task: Did the app display any UI element (text, icon, et al.) in this video from {start_time} to {end_time} suggesting the user shake their phones? List all the timestamp in the format "mm:ss".
        Output: You need to describe the location of the element: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon, while width and height represent its size relative to the video’s width and height, respectively.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=ShakeElementList,
    )

    content = [video, prompt_detect_shake_element]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
