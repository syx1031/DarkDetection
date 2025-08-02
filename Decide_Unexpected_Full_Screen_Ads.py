import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Point import PointLocation_Description
# from Hover import Hover_Description
from Bbox import Location, Bbox_Description


def Decide_Unexpected_Full_Screen_Ads(client, video, recheck_ads_time, click_time_location, voluntary_ad_trigger_element_time_location, start_time, end_time):
    class UnexpectedFullScreenAds(BaseModel):
        unexpected_full_screen_ads: bool = Field(..., description="Whether 'Unexpected Full-Screen Ads' appears or not.")
        click_time: str = Field(...,
                               description="The timestamp when the user clicks on the button which triggers the full-screen ad, must in format 'mm:ss'. If there is no full-screen ad, or if the ad was not triggered by a user action, set this attribute to '00:00'.")
        element_location: Location = Field(..., description="The location of the UI element that triggers the full-screen ad.")
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
            # data['click_or_hover_time'] = self.validate_timestamp(data['click_or_hover_time'])
            # data['ad_start_time'] = self.validate_timestamp(data['ad_start_time'])
            # data['ad_end_time'] = self.validate_timestamp(data['ad_end_time'])
            super().__init__(**data)

    prompt_decide_unexpected_full_screen_ads = f'''
    Context: 
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
        2. Definition of an "Unexpected Full-Screen Ads": This dark pattern occurs when a full-screen ad is displayed to the user in a way that subverts their expectations and without their clear, informed consent. It forces an interruption that the user did not willingly initiate. 
            2.1. This pattern manifests in two primary ways:
                a. Button-Triggering Ads: The user's action (like a click) on a UI element triggers a full-screen ad, but the element's design, text, or context deceptively suggests a different function (e.g., "Next Level", "Continue", a settings icon) and there is no UI element indicating that clicking the button would trigger an ad. The core issue is a mismatch between the element's implied function and its actual ad-triggering outcome.
                b. Unprompted Intrusive Ads: A full-screen ad appears without a direct, immediate, and intentional user action serving as the trigger. This includes ads that appear:
                    - After a period of user inactivity.
                    - Upon app state changes (e.g., finishing a level, loading a new screen).
                    - With a significant delay after a user action, breaking the perceived cause-and-effect link.
            2.2. Clarification on What is NOT a Dark Pattern:
                If a user clicks on a UI element that clearly offers a reward in exchange for watching an ad (e.g., text like "Watch Ad for +100 Coins", "Get Extra Life by Watching", or an icon universally understood to mean "rewarded ad"), this is considered a legitimate, consent-based action, which is NOT an Unexpected Full-Screen Ad.

    Auxiliary information:
        1. Previously, you have identified a complete advertisement during the following time period:
        {recheck_ads_time}
        2. In the few seconds leading up to this ad, you have identified user's clicks at the following timestamps and screen positions. Note that {PointLocation_Description}:
        {click_time_location}
        3. A few seconds before the advertisement appeared, you have identified UI element (text, icon, et al.) that implied or invited the user to watch an ad in the video. Note that {Bbox_Description}:
        {voluntary_ad_trigger_element_time_location}

    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Unexpected Full-Screen Ads" in the ad from {start_time} to {end_time}.
    
    Output: If "Unexpected Full-Screen Ads" occurs, provide the following details:
        - 'click_time': The timestamp "mm:ss" when the user clicks on the button which triggers the full-screen ad.
        - 'element_location': The location of the UI element that triggers the full-screen ad. {Bbox_Description}
        - 'ad_start_time': The time stamp "mm:ss" when the ad starts.
        - 'ad_end_time': The time stamp "mm:ss" when the ad ends.
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