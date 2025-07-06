import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Bbox import Location, Bbox_Description


def Decide_Reward_Based_Ads(client, video, watch_ad_element, reward_element):
    class RewardBasedAds(BaseModel):
        reward_based_ads: bool = Field(..., description="Whether 'Reward-Based Ads' appears or not. ")
        timestamp: str = Field(...,
                               description="The timestamp when the 'Reward-Based Ads' appears, must in format 'mm:ss'.")
        watch_ad_element_location: Location = Field(..., description="The location where the UI element (text, icon, et al.) that implied or invited the user to watch an ad.")
        reward_element_location: Location = Field(..., description="The location where the UI element (text, icon, et al.) that displays rewards to the user.")
        thinking: str = Field(...,
                              description="Provide detailed reasoning about 'Reward-Based Ads'. If it is present, you must explain how the UI elements are combined to form the dark pattern.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    RewardBasedAdsList = [RewardBasedAds]

    prompt_decide_reward_based_ads = f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    2. Reward-Based Ads: Users may be required to watch ads in exchange for other benefits, such as “earning game items” or “unlocking additional features”.

    Auxiliary information:
    1. You previously identified the following UI elements that implied or invited the user to watch an ad in the video. Note that {Bbox_Description}:
    {watch_ad_element}
    2. You previously identified the following UI elements that displays rewards to the user. Note that {Bbox_Description}:
    {reward_element}
    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Reward-Based Ads". Below are common UI element combinations associated with this pattern:
    1. On an app interface, a “watch ad” UI element and a “reward” UI element appear side by side in close proximity. Together, they form a complete semantic unit, informing the user that tapping the button will display an ad and that watching the ad will yield a reward.
    2. On an app interface, a single UI element (such as an icon or text) invites the user to watch an ad and displays the reward that can be obtained by doing so.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=RewardBasedAdsList,
    )

    content = [video, prompt_decide_reward_based_ads]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
