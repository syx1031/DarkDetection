import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Bbox import Location, Bbox_Description


def detect_ad_removal_element_time_location(client, video, start_time=None, end_time=None):
    class AdRemovalElement(BaseModel):
        timestamp: str = Field(...,
                            description="The timestamp at which the 'Ad Removal UI Element' appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(..., description="The location where the 'Ad Removal UI Element' appears.")
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

    AdRemovalElementList = list[AdRemovalElement]

    if start_time and end_time:
        # prompt_detect_ad_removal_element = f'''
        #     Task: Did the app display any UI element (text, icon, et al.) in this video from {start_time} to {end_time} implying that the user can remove ads? List all the timestamp in the format "mm:ss".
        #     Output: You need to describe the timestamp and location of the UI element: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon, while width and height represent its size relative to the video’s width and height, respectively.
        # '''

        prompt_detect_ad_removal_element = f'''
        Context:
            1. You will be analyzing a video segment based on the following information:
                a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
                b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
                c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
            2. Definition of "Ad Removal UI Element": A UI component (like a button, banner, or menu item) that initiates the process of getting an ad-free version of the app, usually through a payment. Look for the following patterns:
                2.1. Direct Text Offers:
                   - UI elements containing explicit keywords such as:
                     - "Remove Ads"
                     - "Go Ad-Free" / "Ad-Free"
                     - "No Ads"
                     - "Disable Ads"
                2.2. Premium/Subscription Upsells:
                   - This is the most common pattern. It involves a Call to Action (CTA) to upgrade to a paid tier.
                   - Firstly, look for CTAs like: "Upgrade to Pro", "Go Premium", "Become a VIP", "Get Plus".
                   - Secondly, check whether this CTA is accompanied by text that lists an ad-free experience as a feature, for example, "Ad-Free experience", "No interruptions", or "No more ads".
                   - The target "Ad Removal UI Element" is the entire element group including a CTA and text that lists an ad-free experience.
                2.3. Specific Iconography:
                   - Icons that symbolize premium status or ad-blocking, when associated with a payment or upgrade offer.
                   - **Common icons:**
                     - **Crown (👑)**: Symbol for "Premium" or "VIP".
                     - **Diamond (💎)**: Symbol for premium features.
                     - **Shield (🛡️)**: Symbol for "blocking" or "protection".
                     - **A crossed-out "AD" symbol**.
                   - These icons may appear standalone or be combined with the text-based offers (from 2.1, 2.2, or text like "One-time purchase", "Buy once, remove ads forever", et al.) When they are presented together, the entire component including both the icon and the text should be considered the 'Ad Removal UI Element'.
                
        Task: Your task is to analyze the video within the period {start_time} to {end_time}, and find out all "Ad Removal UI Elements". Please check each UI element referring to the definitions from 2.1 to 2.3 sequentially.
        
        Output: For each "Ad Removal UI Element" you find, provide the following details. If multiple distinct elements are visible at the same time, list each one separately.
        - `timestamp`: The timestamp "mm:ss" when the element begin to clearly appear.
        - `location`: {Bbox_Description}
        '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=AdRemovalElementList,
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
                types.Part(text=prompt_detect_ad_removal_element)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Detect Ad Removal Text:")
        # print(response.text)
        return json.loads(response.text)
