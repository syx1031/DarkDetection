import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Point import PointLocation


def detect_click_time_location(client, video, start_time=None, end_time=None):
    class Click(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which user click the mouse, accurate to the millsecond and in the format 'mm:ss:xx'.")
        location: PointLocation = Field(..., description="The location where the use clicks.")

        # 自定义验证器确保 period 格式合法
        @classmethod
        def validate_timestamp(cls, value: str) -> str:
            if not re.match(r"^\d{2}:\d{2}:\d{2}$", value):
                raise ValueError("timestamp must be in 'mm:ss:xx' format.")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            # data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    ClickList = list[Click]

    if start_time and end_time:
        prompt_detect_click = '''
        Context:
        1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
            a. The persistent red-bordered circle represents the current position of the cursor. 
            b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.

        Task: Identify the timestamp of each user click, accurate to the millisecond. You need to describe the location of the click: x, y, each ranging from 0 to 1. Here, x and y indicate the relative position of the center of the red circle.
        '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=ClickList,
        )

        start_offset = str(time_to_seconds(start_time)) + 's'
        end_offset = str(time_to_seconds(end_time)) + 's'
        fps = 24

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
                types.Part(text=prompt_detect_click)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Detect Click:")
        # print(response.text)
        return json.loads(response.text)