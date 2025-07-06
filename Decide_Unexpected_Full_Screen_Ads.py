import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Point import PointLocation_Description
from Hover import Hover_Description


def Decide_Unexpected_Full_Screen_Ads(client, video, recheck_ads_time, click_time_location, hover_time_location, watch_ad_icon_time_location, watch_ad_text_time_location, start_time, end_time):
    class UnexpectedFullScreenAds(BaseModel):
        unexpected_full_screen_ads: bool = Field(..., description="Whether 'Unexpected Full-Screen Ads' appears or not. ")
        click_or_hover_time: str = Field(...,
                               description="The timestamp when the user clicks or hovers on the button which triggers the full-screen ad, must in format 'mm:ss'. If there is no full-screen ad, or if the ad was not triggered by a user action, set this attribute to '00:00'.")
        ad_start_time: str = Field(..., description="The timestamp when the full-screen ad starts, must in format 'mm:ss'. If there is no full-screen ad, set this attribute to '00:00'.")
        ad_end_time: str = Field(..., description="The timestamp when the full-screen ad ends, must in format 'mm:ss'. If there is no full-screen ad, set this attribute to '00:00'.")
        thinking: str = Field(...,
                              description="Provide detailed reasoning about 'Unexpected Full-Screen Ads'. If it is present, you must explain how the UI elements are combined to form the dark pattern.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['click_or_hover_time'] = self.validate_timestamp(data['click_or_hover_time'])
            data['ad_start_time'] = self.validate_timestamp(data['ad_start_time'])
            data['ad_end_time'] = self.validate_timestamp(data['ad_end_time'])
            super().__init__(**data)

    prompt_decide_unexpected_full_screen_ads = f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    2. Hover: This describes a sequence of actions where the user clicks and holds the button, then drags it to another place on the screen. Visually, this is reflected by a red circle that contracts and darkens in the center while continuously changing position across consecutive frames, until it returns to original state, indicating the end of the click-and-drag (hover) action.
    3. Unexpected Full-Screen Ads: These ads may manifest in two distinct forms: either triggered by user interaction with a button (denoted as “Button-Triggered Unexpected Ads”), or appearing spontaneously without any user gesture (denoted as “Unprompted Intrusive Ads”).

    Auxiliary information:
    1. The user identified a complete advertisement during the following time period:
    {recheck_ads_time}
    2. In the few seconds leading up to this ad, the user clicked at the following timestamps and screen positions. Note that {PointLocation_Description}
    {click_time_location}
    3. In the few seconds leading up to this ad, the user hovered at the following timestamps and screen positions. Note that {Hover_Description}
    {hover_time_location}
    4. A few seconds before the advertisement appeared, the app displayed the following text implying or suggesting that users should watch ads:
    {watch_ad_text_time_location}
    5. A few seconds before the advertisement appeared, the app displayed the following icon implying or suggesting that users should watch ads:
    {watch_ad_icon_time_location}

    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Unexpected Full-Screen Ads" in the period {start_time}-{end_time}. Below are common UI element combinations associated with this pattern:
    1. Few seconds before the “full-screen ad” appeared, the user performed a “click” or “hover” action, but these actions did not land on or pass over any “watch ad text” or “watch ad icon.” Nevertheless, the “full-screen ad” appeared afterward.
    2. Few seconds before the “full-screen ad” appeared, the user didn't perform any “click” or “hover” action. Nevertheless, the "full-screen ad" appeared afterward.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=UnexpectedFullScreenAds,
    )

    content = [video, prompt_decide_unexpected_full_screen_ads]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    # print("Decide on Unexpected Full-Screen Ads:")
    # print(response.text)
    return json.loads(response.text)