import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Point import PointLocation


def detect_hover_time_location(client, video, start_time=None, end_time=None):
    class Hover(BaseModel):
        start_timestamp: str = Field(...,
                                     description="The timestamp at which user begin the click, accurate to the millsecond and in the format 'mm:ss:xx'.")
        start_location: PointLocation = Field(..., description="The location where the user begin the click.")
        end_timestamp: str = Field(...,
                                   description="The timestamp at which user complete the click, accurate to the millsecond and in the format 'mm:ss:xx'.")
        end_location: PointLocation = Field(..., description="The location where the user end the click.")

        # 自定义验证器确保 period 格式合法
        @classmethod
        def validate_timestamp(cls, value: str) -> str:
            if not re.match(r"^\d{2}:\d{2}:\d{2}$", value):
                raise ValueError("timestamp must be in 'mm:ss:xx' format.")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            # data['start_timestamp'] = self.validate_timestamp(data['start_timestamp'])
            # data['end_timestamp'] = self.validate_timestamp(data['end_timestamp'])
            super().__init__(**data)

    HoverList = list[Hover]

    if start_time and end_time:
        prompt_detect_hover = '''
        Context:
        1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
            a. The persistent red-bordered circle represents the current position of the cursor. 
            b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        2. Hover: This describes a sequence of actions where the user clicks and holds the button, then drags it across the screen. Visually, this is reflected by a red circle that contracts and darkens in the center while continuously changing position across consecutive frames, until it returns to original state, indicating the end of the click-and-drag action.

        Task: Identify the start and end timestamp and location of each user's hover, accurate to the millisecond.
        '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=HoverList,
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
                types.Part(text=prompt_detect_hover)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Detect Hover:")
        # print(response.text)
        return json.loads(response.text)