import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request


def Decide_App_Resumption_Ads(client, video, recheck_ad, outside_interface, start_time, end_time):
    class AppResumptionAds(BaseModel):
        app_resumption_ads: bool = Field(..., description="Whether 'App Resumption Ads' appears or not.")
        start_time: str = Field(...,
                               description="The timestamp when the user left the app, must in format 'mm:ss'. If no ad occurs, set this attribution to '00:00'.")
        ad_start_time: str = Field(...,
                               description="The timestamp when the resumption ad begins, must in format 'mm:ss'. If no ad occurs, set this attribution to '00:00'.")
        end_time: str = Field(..., description="The timestamp when the resumption ad ends, must in format 'mm:ss'. If no ad occurs, set this attribution to '00:00'.")
        thinking: str = Field(...,
                              description="Provide detailed reasoning about 'App Resumption Ads'. If it is present, you must explain how the UI elements are combined to form the dark pattern.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['ad_start_time'] = self.validate_timestamp(data['ad_start_time'])
            data['start_time'] = self.validate_timestamp(data['start_time'])
            data['end_time'] = self.validate_timestamp(data['end_time'])
            super().__init__(**data)

    # prompt_decide_app_resumption_ads = f'''
    # Context:
    # 1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    #     a. The persistent red-bordered circle represents the current position of the cursor.
    #     b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    #     c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    # 2. "App Resumption Ads": When using an app, users may temporarily exit the app by accessing the iPhone’s Control Center or swiping up to return to the Home Screen. Upon returning to the app, they may be forced to view an advertisement before resuming their activity, disrupting their original experience.
    #
    # Auxiliary information:
    # 1. The user may have temporarily left the app and navigated to an external interface during the following time period:
    # {outside_interface}
    # 2. After returning to the app, the user may have been shown an ad immediately during the following time period:
    # {recheck_ad}
    #
    # Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern “App Resumption Ads” in the period {start_time}-{end_time}. Note that only full-screen ads triggered immediately after the user returns to the app qualify as this pattern. Below are common UI element combinations associated with this pattern:
    # 1. The user temporarily left the app to an "external interface", and returned to the app. Within two seconds of returning to the app, the user was immediately shown a "full-screen ad" without having actively clicked on any ad-related content. Note that this "full-screen ad" must not be displayed before the user left the app.
    # '''

    prompt_decide_app_resumption_ads = f'''
    Context: 
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
        2. Definition of an "App Resumption Ads": When using an app, users may temporarily exit the app and visit an external interface (i.e., accessing the iPhone’s Control Center, swiping up to return to the Home Screen, et al.) Upon returning to the app, they may be forced to view an advertisement before resuming their activity, disrupting their original experience.
            2.1. To be classified as "App Resumption Ads", ALL of the following conditions must be met:
                - a. App Departure: The user temporarily navigates away from the app to an `external interface`.
                - b. App Return: The user navigates back to the app from the `external interface`.
                - c. Immediate Full-Screen Ad: Within 2 seconds of the moment that app becomes visible again, a full-screen advertisement appears, completely obscuring the primary content the user was previously interacting with. Additionally, the ad must satisfy following conditions:
                    - Ad is New: The full-screen ad was NOT visible on the screen immediately before a. App Departure.
                    - Ad is Unsolicited: The ad was triggered by the act of returning to the app itself (b. App Return), NOT by an explicit, in-app user action (e.g., clicking a button labeled "Watch Ad for Reward").
                    
    Auxiliary information:
        1. Previously, you identified that the user may have temporarily left the app and navigated to an external interface during the following time period:
        {outside_interface}
        2. Previously, you identified that after returning to the app, the user may have been shown an ad immediately during the following time period:
        {recheck_ad}
        
    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "App Resumption Ads" in the ad from {start_time} to {end_time}.
    
    Output: If "App Resumption Ads" occurs, provide the following details:
        - 'start_time': The timestamp "mm:ss" when the user left the app.
        - 'ad_start_time': The timestamp "mm:ss" when the user returned to the app and resumption ad begins.
        - 'end_time': The timestamp "mm:ss" when the resumption ad ends.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AppResumptionAds,
    )

    content = [video, prompt_decide_app_resumption_ads]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    # print("Decide on App Resumption Ads:")
    # print(response.text)
    return json.loads(response.text)