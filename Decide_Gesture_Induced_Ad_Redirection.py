import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Bbox import Location, Bbox_Description


def Decide_Gesture_Induced(client, video, ad, shake_element):
    class GestureInduced(BaseModel):
        gesture_induced_ad_redirection: bool = Field(..., description="Whether 'Gesture-Induced Ad Redirection' appears or not. ")
        timestamp: str = Field(...,
                               description="The timestamp when the element suggesting users shake their phones appears, must in format 'mm:ss'.")
        close_button_location: Location = Field(..., description="The location where the 'shake phone' element appears.")
        thinking: str = Field(...,
                              description="Provide detailed reasoning about 'Gesture-Induced Ad Redirection'. If it is present, you must explain how the UI elements are combined to form the dark pattern.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            # data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    # GestureInducedList = list[GestureInduced]

    prompt_decide_gesture_induced = f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    2. Gesture-Induced Ad Redirection: The ad interprets subtle user actions as interactions with the ad. For example, when the user slightly shakes the phone, the ad redirects to the landing page. Note that UI element (text, icons, et al.) in the ad that prompt the user to shake the phone can serve as sufficient evidence that the ad uses this mechanism. You do not need to verify whether the user actually shook the phone or whether the ad responded to that action.

    Auxiliary information:
    1. You previously identified the following ad in the video:
    {ad}
    2. You previously identified the following UI element suggesting users shake their phones in this ad. Note that {Bbox_Description}:
    {shake_element}

    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Gesture-Induced Ad Redirection" in the ad from {ad["start_time"]} to {ad["end_time"]}. Below are common UI element combinations associated with this pattern:
    1. In the "ad", the "UI element suggesting users shake phones" appears.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=GestureInduced,
    )

    content = [video, prompt_decide_gesture_induced]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
