import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Bbox import Location, Bbox_Description


def detect_voluntary_ad_trigger_element_time_location(client, video, start_time=None, end_time=None):
    class VoluntaryAdTriggerElement(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which the element (text, icon, et al.) implying or suggesting users to watch ads appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(...,
                                   description="The location where the element implying or suggesting users to watch ads appears.")
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
            data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    VoluntaryAdTriggerElementList = list[VoluntaryAdTriggerElement]

    if start_time and end_time:
        prompt_detect_voluntary_ad_trigger_element = f'''
            Context: 
                1. You will be analyzing a video segment based on the following information:
                    a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
                    b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
                    c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
                2. Definition of a "Voluntary Ad Trigger Element": A "Voluntary Ad Trigger Element" is an interactive UI element (e.g., a button, icon, or text link) that, when clicked by the user, initiates the playback of an ad. The core principle is that the user makes a conscious, voluntary choice to watch the ad, typically in exchange for an in-app reward, benefit, or feature access (this is often called a "Rewarded Ad").
                    2.1 Key Characteristics to Look For (Inclusions). To be identified, a UI element must meet at least one of the criteria ("Explicit Text" or "Suggestive Iconography") below. The "Supporting Context" should be used to help confirm a decision but is not sufficient for identification on its own.
                        - Explicit Text: Look for explicit calls-to-action, such as:
                            "Watch Ad", "Watch Video"
                            "Get Reward", "Claim Reward"
                            "Free Coins", "Get for Free"
                            "Double Reward", "Claim x2"
                            "Unlock by watching a video"
                            "Revive", "+1 Life"
                        - Suggestive Iconography: Look for icons that strongly imply watching a video for a benefit. Common icons include:
                            A play symbol (▶️)
                            A video camera icon (📹)
                            A gift box (🎁), coin stack (🪙), or gem (💎), often with a small play symbol overlayed on it.
                        - Transactional Context: The element is presented as an alternative to paying with in-app currency or real money, or as a way to gain an advantage.
                    2.2 What to Exclude:
                        - Do not identify Ad-in-Progress Indicators. These are labels (e.g., text like "Ad", "Advertisement", "Sponsored", or the AdChoices icon) that appear while an ad is already playing. They label the content, they do not trigger it.
                        - Do not identify standard Banner Ads that are passively displayed on the screen.

            Task: Your task is to analyze the video from {start_time} to {end_time} and find out all "Voluntary Ad Trigger Element". Please check each UI element referring to the definitions from 2.1 to 2.2 sequentially.

            Output: For each "Voluntary Ad Trigger Element" you find, provide the following details. If multiple distinct elements are visible at the same time, list each one separately.
                - `timestamp`: The timestamp "mm:ss" when the element begin to clearly appear.
                - 'location': {Bbox_Description}
        '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=VoluntaryAdTriggerElementList,
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
                types.Part(text=prompt_detect_voluntary_ad_trigger_element)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        return json.loads(response.text)

    else:
        # prompt_detect_watch_ad_element = '''
        #     Task: Did the app display any UI element (text, icon, et al.) in this video that implied or invited the user to watch an ad? List all the timestamp in the format "mm:ss". Note that “watch ad element” refers to UI element indicating to the user clicking on it or a nearby button will lead to watching an ad. It does not refer to icon/text that merely indicates the current content is an ad.
        #     Output: You need to describe the location of the element: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon, while width and height represent its size relative to the video’s width and height, respectively.
        # '''

        prompt_detect_voluntary_ad_trigger_element = f'''
        Context: 
            1. You will be analyzing a video segment based on the following information:
                a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
                b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
                c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
            2. Definition of a "Voluntary Ad Trigger Element": A "Voluntary Ad Trigger Element" is an interactive UI element (e.g., a button, icon, or text link) that, when clicked by the user, initiates the playback of an ad. The core principle is that the user makes a conscious, voluntary choice to watch the ad, typically in exchange for an in-app reward, benefit, or feature access (this is often called a "Rewarded Ad").
                2.1 Key Characteristics to Look For (Inclusions). To be identified, a UI element must meet at least one of the criteria ("Explicit Text" or "Suggestive Iconography") below. The "Supporting Context" should be used to help confirm a decision but is not sufficient for identification on its own.
                    - Explicit Text: Look for explicit calls-to-action, such as:
                        "Watch Ad", "Watch Video"
                        "Get Reward", "Claim Reward"
                        "Free Coins", "Get for Free"
                        "Double Reward", "Claim x2"
                        "Unlock by watching a video"
                        "Revive", "+1 Life"
                    - Suggestive Iconography: Look for icons that strongly imply watching a video for a benefit. Common icons include:
                        A play symbol (▶️)
                        A video camera icon (📹)
                        A gift box (🎁), coin stack (🪙), or gem (💎), often with a small play symbol overlayed on it.
                    - Transactional Context: The element is presented as an alternative to paying with in-app currency or real money, or as a way to gain an advantage.
                2.2 What to Exclude:
                    - Do not identify Ad-in-Progress Indicators. These are labels (e.g., text like "Ad", "Advertisement", "Sponsored", or the AdChoices icon) that appear while an ad is already playing. They label the content, they do not trigger it.
                    - Do not identify standard Banner Ads that are passively displayed on the screen.
                    
        Task: Your task is to analyze the video and find out all "Voluntary Ad Trigger Element". Please check each UI element referring to the definitions from 2.1 to 2.2 sequentially.
        
        Output: For each "Voluntary Ad Trigger Element" you find, provide the following details. If multiple distinct elements are visible at the same time, list each one separately.
            - `timestamp`: The timestamp "mm:ss" when the element begin to clearly appear.
            - 'location': {Bbox_Description}
        '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=VoluntaryAdTriggerElementList,
        )

        content = [video, prompt_detect_voluntary_ad_trigger_element]

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        return json.loads(response.text)
