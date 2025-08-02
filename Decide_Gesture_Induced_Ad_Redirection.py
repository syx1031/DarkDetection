import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Bbox import Location, Bbox_Description


def Decide_Gesture_Induced(client, video, ad, shake_element):
    class GestureInduced(BaseModel):
        gesture_induced_ad_redirection: bool = Field(..., description="Whether 'Gesture-Induced Ad Redirection' appears or not.")
        timestamp: str = Field(...,
                               description="The timestamp when the element suggesting users shake their phones appears, must in format 'mm:ss'.")
        shake_element_location: Location = Field(..., description="The location where the 'shaking phone' UI element appears.")
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
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
        2. Definition of an "Gesture-Induced Ad Redirection": The ad interprets subtle user actions as interactions with the ad. For example, when the user slightly shakes the phone, the ad redirects to the landing page. 
            2.1. The following situation should be identified as an "Gesture-Induced Ad Redirection": 
                - In the ad, the UI element suggesting users shake phones appears. 
                    ** Important: UI element (text, icons, et al.) in the ad that prompt the user to shake the phone can serve as sufficient evidence that the ad uses this mechanism. You do not need to verify whether the user actually shook the phone or whether the ad responded to that action.

    Auxiliary information:
        1. You previously identified the following ad in the video:
        {ad}
        2. You previously identified the following UI element suggesting users shake their phones in this ad. Note that {Bbox_Description}:
        {shake_element}

    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Gesture-Induced Ad Redirection" in the ad from {ad["start_time"]} to {ad["end_time"]}.
    
    Output: If "Gesture-Induced Ad Redirection" occurs, provide the following details:
        - 'timestamp': The timestamp "mm:ss" when the 'shaking phone' UI element appears.
        - 'shake_element_location': The location where the 'shaking phone' UI element appears. You can copy it from Auxiliary information 2.
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
