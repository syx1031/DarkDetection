import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Bbox import Bbox_Description


def Decide_Ad_Without_Exit_Option(client, video, ad, close_button):
    class AdWithoutExitOption(BaseModel):
        ad_without_exit_option: bool = Field(..., description="Whether 'Ad Without Exit Option' appears or not. ")
        timestamp: str = Field(...,
                               description="The timestamp when the first close button appears, must in format 'mm:ss'. If there is no close button, please fill '00:00'.")
        thinking: str = Field(...,
                              description="Provide detailed reasoning about 'Ad Without Exit Option'. If it is present, you must explain how the UI elements are combined to form the dark pattern.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            # data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    prompt_decide_ad_without_exit_option = f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    2. Ad Without Exit Option: Some ads do not provide a close button or delay their appearance to force users to watch them. The absence of a close button during the first three seconds of ad playback also constitutes this type of dark pattern.

    Auxiliary information:
    1. You previously identified the following ad in the video:
    {ad}
    2. You previously identified the following ad close buttons in the ad. Note that {Bbox_Description}:
    {close_button}

    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Ad Without Exit Option" in the ad from {ad["start_time"]} to {ad["end_time"]}. Below are common UI element combinations associated with this pattern:
    1. There is no "close button" present on the "non-full-screen ad" interface.
    2. No "close button" appears within the first three seconds of the "full-screen ad" display.
    3. No "close button" appears at any point during the entire "full-screen ad" display.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdWithoutExitOption,
    )

    content = [video, prompt_decide_ad_without_exit_option]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
