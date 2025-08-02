import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Bbox import Bbox_Description


def Decide_Paid_Ad_Removal(client, video, purchase_interface, ad_removal_element, start_time=None, end_time=None):
    class PaidAdRemoval(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp when the purchase interface appears, should be represented in the format 'mm:ss'.")
        paid_ad_removal: bool = Field(..., description="Whether 'Paid Ad Removal' appears or not.")
        thinking: str = Field(...,
                              description="Provide detailed reasoning about 'Paid Ad Removal'. If it is present, you must explain how the UI elements are combined to form the dark pattern.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            # data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    if start_time and end_time:
        # prompt_decide_paid_ad_removal = f'''
        # Context:
        # 1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        #     a. The persistent red-bordered circle represents the current position of the cursor.
        #     b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        #     c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
        # 2. "Paid Ad Removal": Some apps offer a paid option to remove ads.
        #
        # Auxiliary information:
        # 1. A purchase interface may have appeared at the following timestamp:
        # {purchase_interface}
        # 2. UI element (text, icon, et al.) that implied or invited the user to remove ads have appeared at the following timestamp and location. Note that {Bbox_Description}:
        # {ad_removal_element}
        #
        # Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern “Paid Ad Removal” during {start_time}-{end_time}. Below are common UI element combinations associated with this pattern:
        # 1. The "purchase interface" displayed "ad removal text" or "ad removal icon", indicating to the user that ads would be removed after making a purchase.
        # 2. The app displayed "ad removal text" or "ad removal icon", and when the user clicked on it, a "purchase interface" appeared, indicating that a purchase was required to remove ads.
        # '''

        prompt_decide_paid_ad_removal = f'''
        Context: 
            1. You will be analyzing a video segment based on the following information:
                a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
                b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
                c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
            2. Definition of "Paid Ad Removal": It is defined as any feature or option within the app that allows the user to pay money in exchange for removing advertisements.
                2.1. The presence of this dark pattern is confirmed by one of the following scenarios:
                    - Direct Purchase Interface: A purchase interface appears on the screen, and it contains text or icons explicitly offering ad removal (e.g., "Remove Ads," "Go Ad-Free," "No Ads," "Ad-Free Version").
                    - Triggered Purchase Interface: The user clicks on a UI element (like a button, banner, or menu item) that offers ad removal, and this action subsequently triggers a purchase interface.
                    
        Auxiliary information:
            1. Previously, you identified that these purchase interfaces may have appeared:
            {purchase_interface}
            2. Previously, you identified that these UI elements (text, icon, et al.) that invited the user to remove ads have appeared at the following timestamp and location. Note that {Bbox_Description}:
            {ad_removal_element}
                    
        Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern “Paid Ad Removal” during {start_time}-{end_time}.
        
        Output: If "Paid Ad Removal" occurs, provide the following details:
            - 'timestamp': The timestamp "mm:ss" when the purchase interface appears.
        '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=PaidAdRemoval,
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
                types.Part(text=prompt_decide_paid_ad_removal)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Decide on Paid Ad Removal:")
        # print(response.text)
        return json.loads(response.text)
