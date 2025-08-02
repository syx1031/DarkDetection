import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Bbox import Location, Bbox_Description


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
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the red circle contracts and its center turns black.
            
        2. Ad Close Button: An ad close button refers to any button that visually appears to allow the user to exit the ad when clicked. These buttons are typically located in a corner of the ad and are often shaped like an “X” or resemble a video fast-forward icon. Some close buttons may include text labels such as “Skip Video.” Any such design shown during the ad playback should be considered a close button, regardless of whether clicking it actually closes the ad.

        Task: Did the ad during {start_time}-{end_time} displays any close button? List all the timestamp it occurs in the format "mm:ss".
    '''

    prompt_detect_close_button = f'''
        Context:
            1. You will be analyzing a video segment based on the following information:
                a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
                b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
                c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
            2. Definition of "Ad Close Buttons": An "Ad Close Button" is any clickable UI element that visually suggests to the user that they can dismiss, skip, or otherwise exit the advertisement. 
            - Your judgment must be based solely on the button's visual appearance and its implied promise to the user. The button's actual function when clicked is irrelevant. Even if a button is deceptive (e.g., it leads to the App Store instead of closing the ad), it should be identified as a close button if it looks like one. These buttons fall into several categories:
                2.1. Classic Close Icons: Standard icons that directly signal "close."
                    Examples: An "X" icon (e.g., X, ✕, ⊗).
                2.2. Text-Based Buttons: Buttons that use words to indicate the action.
                    Examples: Buttons with text labels like "Close," "Skip," "Skip Ad," or "Skip Video." This also includes buttons that appear after a countdown (e.g., the text changes from "You can skip in 3s" to "Skip Ad").
                2.3. Progression or Skip Icons: Icons that metaphorically suggest moving past the ad.
                    Examples: A "fast-forward" icon (>> or ►►), which implies speeding through the ad. A "next" icon (>), which implies moving on to the next content and sometimes combines with text "Continue to the app."
                2.4. Other Subtle Variations: Any other design that clearly implies an exit.
            
        Task: Your task is to analyze the video within the period {start_time} to {end_time}, and find out all "Ad Close Buttons". Please check each UI element referring to the definitions from 2.1 to 2.4 sequentially.
        
        Output: For each "Ad Close Button" you find, provide the following details. If multiple distinct elements are visible at the same time, list each one separately.
        - `timestamp`: The timestamp "mm:ss" when the element begin to clearly appear.
        - `location`: {Bbox_Description}
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
