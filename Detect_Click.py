import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Point import PointLocation, PointLocation_Description


def detect_click_time_location(client, video, start_time=None, end_time=None):
    class Click(BaseModel):
        start_timestamp: str = Field(...,
                               description="The timestamp at which user starts to click the mouse, accurate to the millsecond and in the format 'mm:ss:xx'.")
        end_timestamp: str = Field(...,
                                     description="The timestamp at which user finishes to click the mouse, accurate to the millsecond and in the format 'mm:ss:xx'.")
        start_location: PointLocation = Field(..., description="The location where the use starts the click.")
        end_location: PointLocation = Field(..., description="The location where the use finishes the click.")

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
        # prompt_detect_click = '''
        # Context:
        # 1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        #     a. The persistent red-bordered circle represents the current position of the cursor.
        #     b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        #
        # Task: Identify the timestamp of each user click, accurate to the millisecond. You need to describe the location of the click: x, y, each ranging from 0 to 1. Here, x and y indicate the relative position of the center of the red circle.
        # '''

        prompt_detect_click = f'''
            Context:
                1. Video: You will be analyzing a video file that is a screen recording of a user interacting with an iPhone application:
                    a. Video Source: The video is a screen recording of an iPhone app.
                    b. Cursor Representation: A persistent red-bordered circle on the screen represents the real-time position of the user's mouse cursor.
                2. Click:
                    a. Click Indication: A "click" action is visually indicated by a specific, temporary change in the cursor's appearance. The cursor translates from a red circle to a yellow square.
                    b. Click Definition: A single, complete click event starts at the exact moment the yellow square appears, and ends at the exact moment it returns to the red-bordered circle.
                    
            Task: Your mission is to meticulously analyze the provided video from {start_time} to {end_time}, tracking the cursor's state frame-by-frame to identify every single click event. For each event, you must log its precise temporal and spatial data. Detailed Instructions:
                Step 1. Continuous Tracking: Scan the video from beginning to end. In every frame, you must locate the cursor (red-bordered circle or yellow square) and be aware of its position (as X, Y coordinates) and its size/appearance.
                Step 2. Event Detection: Pay extremely close attention to subtle changes. You are looking for the transition from the "normal state" (red-bordered circle) to the "clicked state" (yellow square) and back. The changes in the cursor's shape and color can be very subtle and easy to miss. Therefore, it is critical that you perform a meticulous, frame-by-frame comparison, contrasting each frame with the previous one, to reliably detect these transitions.
                    
            Output: For each click you find, provide the following details:
                - start_time: The timestamp "mm:ss:xx" when the click event begins (circle starts to contract/change color).
                - end_time: The timestamp "mm:ss:xx" when the click event ends (circle returns to its normal state).
                - start_position: The (x, y) coordinates of the center of the yellow square at start_time. {PointLocation_Description}
                - end_position: The (x, y) coordinates of the center of the yellow square at end_time. (Note: For a simple click, this will likely be identical to the start_position. For a click-and-drag, it might differ, but based on the definition, we are treating a single press-and-release as one click). {PointLocation_Description}
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
