import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, generate_part
from rag import load_database, generate_video_summarize, generate_exemplars_parts


class AdSegment(BaseModel):
    start_timestamp: str = Field(..., description="The timestamp at which the ad appear: should be represented in the format 'mm:ss'.")
    end_timestamp: str = Field(..., description="The timestamp at which the ad end: should be represented in the format 'mm:ss'.")
    full_screen: bool = Field(..., description="Whether the ad is a full-screen ad at the beginning of it.")
    description: str = Field(..., description="Briefly describe the ad’s position and content in one to two sentences.")
    thinking: str = Field(..., description="Explain in detail the reasoning behind your judgment that an advertisement appeared during this time period.")

    # 自定义验证器确保 period 格式合法
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        if not re.match(r"^\d{2}:\d{2}$", value):
            raise ValueError("timestamp must be in 'mm:ss' format.")
        return value

    def __init__(self, **data):
        # 手动调用验证器
        data['start_timestamp'] = self.validate_timestamp(data['start_timestamp'])
        data['end_timestamp'] = self.validate_timestamp(data['end_timestamp'])
        super().__init__(**data)


# 用于整体结构的 list 类型
AdSegmentList = list[AdSegment]


def detect_ads(client, video):
    prompt_detect_ads = '''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

    Task: 
    1. Detect Full-Screen Ads:
        1.1. At what time did full-screen advertisements appear (use the format xx:xx–xx:xx)? List all time periods during which full-screen ads appeared. Note: (1) If the ad occupies the entire screen or the majority, it should be classified as a "full-screen" ad; conversely, if the ad does not start in full-screen but later becomes full-screen, it should not be classified as a "full-screen ad". (2) Although App Store pages or prompt pages requesting users to rate the app can be part of an ad, they are not considered ad interfaces when standalone. (3) The interface for paid ad removal is part of the app’s functional UI and does not count as an ad. (4) Interfaces that merely request the user to watch an ad (e.g., “Watch an ad for...”) are not considered ads.
        1.2. Reconsider the ad time intervals you identified in Task 1---especially those with similar content or consecutive time intervals---and determine whether they represent different stages of the same ad (e.g., a video followed by an playable demo) or different UI components of the same stage (e.g., a full-screen video and a banner with summary information below it). Then treat these intervals as a single ad for further analysis. List all time periods during which ads appeared again after your reconsideration.
    2. Detect Non-Full-Screen Ads:
        2.1 At what time did non-full-screen ads appear (use the format xx:xx-xx:xx)? List all time periods. Note: (1) You need to examine the video frame by frame, paying special attention to the top and bottom of each frame to check for the presence of banner ads. These ads are typically characterized by a width that is close to or equal to the frame width and contain advertising content that is clearly different from the content of the current app. (2) The start time of a non-fullscreen ad refers to the moment when the ad content pops up over the app interface in the background. The end time is the moment when the ad content disappears. An ad is considered to have ended when one of the following occurs: the underlying app interface is revealed again, the ad space is replaced by a feedback interface from the ad platform (e.g., “Ads by Google”), or the original ad area becomes an empty, contentless placeholder. (3) Non-fullscreen ads may scroll through multiple pieces of content or switch the ad content within the same placement. These content changes should not be considered as the start or end of an ad. In other words, you should not treat a single continuous ad display as multiple ads simply because the content changes.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdSegmentList,
        topP=0.01,
        topK=1,
        temperature=0.0,
    )
    contents = [video, prompt_detect_ads]

    # Send request with function declarations
    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        # contents=[video_file, prompt],
        contents=contents,
        config=config,
    )

    # print("Detect Ad:")
    # print(response.text)
    return json.loads(response.text)


retriever = load_database()


def recheck_ads(client, video, start_time, end_time):
    class AdAttribution(BaseModel):
        start_time: str = Field(...,
                                description=f"The timestamp when the ad starts, should be represented in the format 'mm:ss'. If no ad occurs in provided period, set this attribution to '00:00'.")
        end_time: str = Field(...,
                              description=f"The timestamp when the ad ends, should be represented in the format 'mm:ss'. If no ad occurs in provided period, set this attribution to '00:00'.")
        full_screen: bool = Field(..., description="Whether the ad is a full-screen ad at the beginning of it.")
        thinking: str = Field(..., description="Provide detailed analyze about above attributions of the ad.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['start_time'] = self.validate_timestamp(data['start_time'])
            data['end_time'] = self.validate_timestamp(data['end_time'])
            super().__init__(**data)

    prompt_recheck_ad = f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

    Task: An ad may be displayed in the video during the period {start_time}-{end_time}. Please review this period and verify the following attributions of the ad:
    1. Check the few seconds before {start_time} to determine whether the preceding content also belongs to the same ad. Adjust the ad’s start time accordingly based on your analysis. A common scenario is that the app initially displays a non-full-screen ad interface (which contains ad content rather than merely prompting the user to watch an ad). Subsequently, due to user gestures or automatic transitions, the ad switches to a full-screen interface.
    2. Check the few seconds after {end_time} to determine whether the subsequent content also belongs to the same ad. Adjust the ad’s end time accordingly based on your analysis. 
    3. Check the beginning of the ad. If the ad occupies the entire screen or the majority, it should be classified as a "full-screen ad"; conversely, if the ad does not start in full-screen but later becomes full-screen, it should not be classified as a "full-screen ad".
    '''

    ad_summarize = generate_video_summarize(client, video, start_time, end_time)
    retriever_result = retriever.invoke(ad_summarize)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdAttribution,
        topP=0.01,
        topK=1,
        temperature=0.0,
    )

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=types.Content(
            parts=[generate_part(video.uri, "v"), generate_part(prompt_recheck_ad, "t")] + generate_exemplars_parts(retriever_result),
        ),
        config=config,
    )

    # print("Recheck Ad:")
    # print(response.text)
    return json.loads(response.text), ad_summarize, retriever_result