import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Bbox import Location


def detect_watch_ad_element_time_location(client, video, start_time=None, end_time=None):
    class WatchAdElement(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which the element (text, icon, et al.) implying or suggesting users to watch ads appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(...,
                                   description="The location where the element implying or suggesting users to watch ads appears.")
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
            data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    WatchAdElementList = list[WatchAdElement]

    prompt_detect_watch_ad_element = '''
        Task: Did the app display any UI element (text, icon, et al.) in this video that implied or invited the user to watch an ad? List all the timestamp in the format "mm:ss". Note that “watch ad element” refers to UI element indicating to the user clicking on it or a nearby button will lead to watching an ad. It does not refer to icon/text that merely indicates the current content is an ad.
        Output: You need to describe the location of the element: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon, while width and height represent its size relative to the video’s width and height, respectively.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=WatchAdElementList,
    )

    content = [video, prompt_detect_watch_ad_element]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
