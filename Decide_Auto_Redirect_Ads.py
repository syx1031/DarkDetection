import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Point import PointLocation_Description


def Decide_Auto_Redirect_Ads(client, video, ad, landing_page_time, click_time_location):
    class AutoRedirectAds(BaseModel):
        auto_redirect_ads: bool = Field(..., description="Whether 'Auto-Redirect Ads' appears or not. ")
        timestamp: str = Field(...,
                               description="The timestamp when the landing page appears, must in format 'mm:ss'.")
        thinking: str = Field(...,
                              description="Provide detailed reasoning about 'Auto-Redirect Ads'. If it is present, you must explain how the UI elements are combined to form the dark pattern.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    # prompt_decide_auto_redirect_ads = f'''
    # Context:
    # 1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    #     a. The persistent red-bordered circle represents the current position of the cursor.
    #     b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    #     c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    # 2. Auto-Redirect Ads: An ad automatically redirects to its landing page during the ad display, without the user clicking any button in the ad.
    #
    # Auxiliary information:
    # 1. You previously identified the following ad in the video:
    # {ad}
    # 2. You previously identified the following landing page in the ad:
    # {landing_page_time}
    # 3. You previously identified the following clicks by the user just before the landing page appearing. Note that {PointLocation_Description}:
    # {click_time_location}
    # Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Auto-Redirect Ads" in the ad from {ad["start_time"]} to {ad["end_time"]}. Below are common UI element combinations associated with this pattern:
    # 1. In the "ad", the user did not "click" on any content within the ad, yet the ad automatically redirected to the "landing page".
    # '''

    prompt_decide_auto_redirect_ads = f'''
    Context: 
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
        2. Definition of an "Auto-Redirect Ad": An "Auto-Redirect Ad" is a dark pattern where an advertisement automatically navigates the user to a landing page (e.g., a website or an app store page) without a clear, direct, and intentional action from the user to trigger that navigation.
            2.1. The following situation should be identified as an "Auto-Redirect Ad":
                - The ad redirects to the landing page with no user clicks occurring within the ad's display area beforehand.
                - The ad redirects immediately upon being displayed, giving the user no time to interact.
                - The ad features a countdown timer, and upon reaching zero, it automatically redirects instead of closing or allowing the user to proceed.
            2.2. Exclusion Criteria (This IS NOT an "Auto-Redirect Ad" if):
                - The user clicks on any element within the ad's frame just before the redirect occurs. Even if the user clicks a deceptive element (like a fake "close" button) or an apparently non-interactive area that causes a redirect, it is considered a user-initiated action.
                - The user clicks on a clear Call-to-Action (CTA) button, such as "Learn More," "Download," "Play Now," etc.
                
    Auxiliary information:
        1. You previously identified the following ad in the video:
        {ad}
        2. You previously identified the following landing page in the ad:
        {landing_page_time}
        3. You previously identified the following clicks by the user just before the landing page appearing. Note that {PointLocation_Description}:
        {click_time_location}
        
    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Auto-Redirect Ads" in the ad from {ad["start_time"]} to {ad["end_time"]}.
    
    Output: If "Auto-Redirect Ads" occurs, provide the following details:
        - 'timestamp': The timestamp "mm:ss" when the landing page appears.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AutoRedirectAds,
    )

    content = [video, prompt_decide_auto_redirect_ads]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
