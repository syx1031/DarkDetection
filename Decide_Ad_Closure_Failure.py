import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Bbox import Location
from Point import PointLocation_Description
from Bbox import Bbox_Description
from typing import Literal


def Decide_Ad_Closure_Failure(client, video, ad, close_button, click):
    class AdClosureFailure(BaseModel):
        ad_closure_failure: bool = Field(..., description="Whether 'Ad Closure Failure' appears or not. ")
        timestamp: str = Field(...,
                               description="The timestamp when the user clicks the close button, must in format 'mm:ss'.")
        close_button_location: Location = Field(..., description="The location where the close button appears.")
        manifestation: Literal["Multi-Step Ad Closure", "Closure Redirect Ads", "Forced Ad-Free Purchase Prompts", "Others"] = Field(..., description="Determine the specific manifestation of 'ad closure failure' based on what happens after the user clicks the close button, must in 'Multi-Step Ad Closure', 'Closure Redirect Ads', or 'Forced Ad-Free Purchase Prompts', or 'Others'.")
        thinking: str = Field(...,
                              description="Provide detailed reasoning about 'Ad Closure Failure'. If it is present, you must explain how the UI elements are combined to form the dark pattern.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    AdClosureFailureList = [AdClosureFailure]

    prompt_decide_ad_closure_failure = f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    2. Ad Closure Failure: After clicking the close button, an ad may fail to close as expected. There are several more detailed manifestations:
        a. "Multi-Step Ad Closure": requires users to complete multiple dismissal actions, as the initial close button click merely redirects to another advertisement page.
        b. "Closure Redirect Ads":  immediately direct users to promotional landing pages, typically app store destinations, upon closure attempts.
        c. "Forced Ad-Free Purchase Prompts": present subscription offers immediately after ad dismissal, effectively transforming the closure action into a monetization opportunity. 

    Auxiliary information:
    1. You previously identified the following ad in the video:
    {ad}
    2. You previously identified the following ad close buttons in the ad. Note that {Bbox_Description}
    {close_button}
    3. You previously identified the following clicks by the user. Note that {PointLocation_Description}:
    {click}
    
    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Ad Closure Failure" in the ad from {ad["start_time"]} to {ad["end_time"]}. Below are common UI element combinations associated with this pattern:
    1. In the "ad", the user "click" the "close button". However, the "ad" did not close and instead displayed another interface. (Multi-Step Ad Closure)
    2. In the "ad", the user "click" the "close button". However, the "ad" redirects to the landing page. (Closure Redirect Ads)
    3. In the "ad", the user "click" the "close button". However, the app displayed an interface requiring the user to pay in order to close the ad. (Forced Ad-Free Purchase Prompts)
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdClosureFailureList,
    )

    content = [video, prompt_decide_ad_closure_failure]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
