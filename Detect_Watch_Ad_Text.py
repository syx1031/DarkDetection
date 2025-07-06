import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Bbox import Location


def detect_watch_ad_text_time_location(client, video, start_time=None, end_time=None):
    class WatchAdText(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which the text implying or suggesting users to watch ads appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(...,
                                   description="The location where the text implying or suggesting users to watch ads appears.")
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

    WatchAdTextList = list[WatchAdText]

    if start_time and end_time:
        prompt_detect_watch_ad_text = '''
            Task: Did the app display any text in this video that implied or invited the user to watch an ad? List all the timestamp in the format "mm:ss". Note that “watch ad text” refers to text indicating to the user clicking on it or a nearby button will lead to watching an ad. It does not refer to text that merely indicates the current content is an ad.
            Output: You need to describe the location of the text box: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the text box, while width and height represent its size relative to the video’s width and height, respectively.
            '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=WatchAdTextList,
        )

        start_offset = str(time_to_seconds(start_time)) + 's'
        end_offset = str(time_to_seconds(end_time)) + 's'
        fps = 5

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
                types.Part(text=prompt_detect_watch_ad_text)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Detect Watch Ad Text:")
        # print(response.text)
        return json.loads(response.text)