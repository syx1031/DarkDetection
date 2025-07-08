import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Bbox import Location


def detect_close_button_time_location(client, video, start_time=None, end_time=None):
    class CloseButton(BaseModel):
        close_button: bool = Field(..., description="Decide whether a 'close button' occurs or not.")
        timestamp: str = Field(...,
                               description="The timestamp at which the close button appears in the ad. Should be represented in the format 'mm:ss'. If no close button occurs, please fill '00:00'.")
        location: Location = Field(..., description="The location where the close button appears in the ad.")
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

    CloseButtonList = list[CloseButton]

    prompt_detect_close_button = f'''
        Context: 
        1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
            a. The persistent red-bordered circle represents the current position of the cursor. 
            b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
            c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
        2. Ad Close Button: An ad close button refers to any button that visually appears to allow the user to exit the ad when clicked. These buttons are typically located in a corner of the ad and are often shaped like an “X” or resemble a video fast-forward icon. Some close buttons may include text labels such as “Skip Video.” Any such design shown during the ad playback should be considered a close button, regardless of whether clicking it actually closes the ad.

        Task: Did the ad during {start_time}-{end_time} displays any close button? List all the timestamp it occurs in the format "mm:ss".
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=CloseButtonList,
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
            types.Part(text=prompt_detect_close_button)
        ]
    )

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
