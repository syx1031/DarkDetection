import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Bbox import Location


def detect_reward_element_time_location(client, video, start_time=None, end_time=None):
    class RewardElement(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which the element (text, icon, et al.) displaying rewards to the user, such as unlocking more app features or offering in-game currency or items. Should be represented in the format 'mm:ss'.")
        location: Location = Field(...,
                                   description="The location where the element displaying rewards appears.")
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

    RewardElementList = list[RewardElement]

    prompt_detect_reward_element = '''
        Task: Did the app display any UI element (text, icon, et al.) in this video that displays rewards to the user, such as unlocking more app features or offering in-game currency or items? List all the timestamp in the format "mm:ss". Note that "rewards" may include in-game currency, items, doubling game earnings, or unlocking new features in the app or game.
        Output: You need to describe the location of the element: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon, while width and height represent its size relative to the video’s width and height, respectively.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=RewardElementList,
    )

    content = [video, prompt_detect_reward_element]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
