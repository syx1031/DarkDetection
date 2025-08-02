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
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
            
    Task: 
    1. Detect Full-Screen Ads:
        1.1. At what time did full-screen advertisements appear (use the format xx:xx–xx:xx)? List all time periods during which full-screen ads appeared. Note: 
            (1) If the ad occupies the entire screen or the majority, it should be classified as a "full-screen" ad; conversely, if the ad does not start in full-screen but later becomes full-screen, it should not be classified as a "full-screen ad". 
            (2) Although App Store pages or prompt pages requesting users to rate the app can be part of an ad, they are not considered ad interfaces when standalone. 
            (3) The interface for paid ad removal is part of the app’s functional UI and does not count as an ad. 
            (4) Interfaces that merely request the user to watch an ad (e.g., “Watch an ad for...”) are not considered ads.
        1.2. Reconsider the ad time intervals you identified in Task 1---especially those with similar content or consecutive time intervals---and determine whether they represent different stages of the same ad (e.g., a video followed by an playable demo) or different UI components of the same stage (e.g., a full-screen video and a banner with summary information below it). Then treat these intervals as a single ad for further analysis. List all time periods during which ads appeared again after your reconsideration.
    2. Detect Non-Full-Screen Ads:
        2.1 At what time did non-full-screen ads appear (use the format xx:xx-xx:xx)? List all time periods. Note: 
            (1) You need to examine the video frame by frame, paying special attention to the top and bottom of each frame to check for the presence of banner ads. These ads are typically characterized by a width that is close to or equal to the frame width and contain advertising content that is clearly different from the content of the current app. 
            (2) The time periods for non-full-screen ads and full-screen ads are mutually exclusive and cannot overlap. Sometimes, a full-screen ad (e.g., a video or demo) will display a smaller banner in one of its corners. This banner must be treated as part of the full-screen ad, not as a separate non-full-screen ad. When you are locating non-full-screen ads, be careful to distinguish them from these in-ad banners.
            (3) The start time of a non-full-screen ad refers to the moment when the ad content pops up over the app interface in the background. The end time is the moment when the ad content disappears. An ad is considered to have ended when one of the following occurs:
                - The underlying app interface is revealed again
                - The ad space is replaced by a feedback interface from the ad platform (e.g., “Ads by Google”)
                - The original ad area becomes an empty, contentless placeholder. 
            (4) Non-full-screen ads may scroll through multiple pieces of content or switch the ad content within the same placement. These content changes should not be considered as the start or end of an ad. In other words, you should not treat a single continuous ad display as multiple ads simply because the content changes.
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


def recheck_ads(client, video, start_time, end_time, full_screen, ad_time):
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

    # prompt_recheck_ad = f'''
    # Context:
    # 1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    #     a. The persistent red-bordered circle represents the current position of the cursor.
    #     b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    #     c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    #
    # Task: An ad may be displayed in the video during the period {start_time}-{end_time}. Please review this period and verify the following attributions of the ad:
    # 1. Determine whether there is an ad in the period from {start_time} to {end_time}.
    # 2. If there is an ad from {start_time} to {end_time}, then check the few seconds before {start_time} to determine whether the preceding content also belongs to the same ad. Adjust the ad’s start time accordingly based on your analysis.
    #     Example: If the app initially displays a non-full-screen interface that contains actual advertisement content (as opposed to simply prompting the user to watch an ad), and later—either through user interaction or automatic transition—it switches to a full-screen ad, you should consider the start time to be when the non-full-screen ad first appears.
    # 2. If there is an ad from {start_time} to {end_time}, then check the few seconds after {end_time} to determine whether the subsequent content also belongs to the same ad. Adjust the ad’s end time accordingly based on your analysis.
    # 3. If there is an ad from {start_time} to {end_time}, check whether the ad is a "full-screen" ad.
    #     Example: If the ad occupies the entire screen or the majority at the beginning of its display, you should consider as a "full-screen ad".
    #     Counterexample: If the ad does not start in full-screen but later becomes full-screen, it should be classified as a "non-full-screen ad".
    # '''

    prompt_recheck_ad = f'''
    Context:
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
        2. Full-Screen Ads: If the ad occupies the entire screen or the majority, it should be classified as a "full-screen" ad; conversely, if the ad does not start in full-screen but later becomes full-screen, it should not be classified as a "full-screen ad". 
            (1) Although App Store pages or prompt pages requesting users to rate the app can be part of an ad, they are not considered ad interfaces when standalone. 
            (2) The interface for paid ad removal is part of the app’s functional UI and does not count as an ad. 
            (3) Interfaces that merely request the user to watch an ad (e.g., “Watch an ad for...”) are not considered ads.
        3. Non-Full-Screen Ads: If the ad doesn't occupy the entire screen or the majority at the beginning, it should be classified as a "non-full-screen" ad.
            (1) These ads are typically characterized by a width that is close to or equal to the frame width and contain advertising content that is clearly different from the content of the current app. 
            (2) The start time of a non-full-screen ad refers to the moment when the ad content pops up over the app interface or banner in the background. The end time is the moment when the ad content disappears. An ad is considered to have ended when one of the following occurs: 
                - The underlying app interface is revealed again
                - The ad space is replaced by a feedback interface from the ad platform (e.g., “Ads by Google”)
                - The original ad area becomes an empty, contentless placeholder. 
            (3) Non-full-screen ads may scroll through multiple pieces of content or switch the ad content within the same placement. These content changes should not be considered as the start or end of an ad. In other words, you should not treat a single continuous ad display as multiple ads simply because the content changes.
            
    Auxiliary Information: 
        1. Previously, you identified a {"full-screen" if full_screen else "non-full-screen"} ad in the period {start_time}-{end_time}:
        {ad_time}
        
    Task: Your task is to reanalyze the advertisement within the period {start_time} to {end_time}. Please follow these steps sequentially and provide a clear analysis for each.
        Step 1: Ad Identification.
            First, use your prior thinking to find the specified ad between {start_time} and {end_time}. If it is not found, state that and stop here. The following steps only apply if an ad is identified.
        Step 2: Differentiate Between App Content and Ad Content.
            To establish a clear boundary between the app being used and the advertisement, perform the following analysis:
                - Analyze the App: Observe the entire video and summarize the primary content and functionality of the app being used.
                - Analyze the Advertisement: Observe the current advertisement and summarize the product or service being promoted.
                    - If the advertised product is another app: Summarize that app's main content and functionality.
                    - If the advertised product is not an app (e.g., a physical item, a service): Summarize the key information presented about it.
            Clarify the Boundary: Based on the analysis above, you must be able to clearly distinguish between the app being used and the advertised product based on their respective content and purpose.
        Step 3: Determine the Ad's Precise Start and End Time.
            - A segment of the video is considered part of an ad if it meets one of these criteria:
                - The displayed content or UI design is consistent with the advertisement, not the host app (Please refer to clues from Step 2). Warning: Ads may feature playable demos. Do not confuse these demos with the host app's actual game. You could distinguish them by comparing the gameplay of the demo versus the gameplay of the host app.
                - The frame contains an ad-indicator icon (e.g., ⓘ, or "AdChoice" icon) in a corner.
                - Interfaces that are displayed to present suitable ad content, including those asking for the user's age, or soliciting information or obtaining consent from the user.
            - An ad segment is considered connected with the ad you identified if they are linked by one of the following:
                - A seamless transition on the video timeline.
                - A brief loading screen (e.g., a black or white screen) between them.
                - Another interface belonging to the ad, such as a landing page.
                - A transition through a system UI out of the app (e.g., Home Screen, App Switcher, Notification Center, Control Center).
            
            You must follow these steps iteratively to finalize the ad's precise start time.
                Step 3.1: Refine the Current Start Time. For your current determined ad start time, perform the following two checks in order:
                    (A) Look Backwards: Examine the preceding several seconds (e.g., 10-15 seconds) before your current identified start time. Is this preceding segment also ad content, AND is it connected to your current ad start time?
                        If both conditions are true, you should update the start time to the beginning of this newly found, earlier segment.
                    (B) Sanity-Check Forwards (Only if A is false): If you did not update the start time in the step above, re-evaluate the content immediately after your current start time. Is this segment, previously considered part of the ad, actually host app content?
                        If yes, you must adjust the start time to the correct moment the ad truly begins.
                Step 3.2: Iterate or Finalize
                If you revised the start time in Step 3.1 (either backward or forward), you should repeat the entire process from Step 3.1 with the new, updated start time.
                
                Step 3.3: Refine the Current End Time. For your current determined ad end time, perform the following two checks in order:
                    (A) Look Forwards: Examine the subsequent several seconds (e.g., 10-15 seconds) after your current identified end time. Is this subsequent segment also ad content, AND is it connected to your current ad end time?
                        If both conditions are true, you should update the end time to the ending of this newly found, subsequent segment.
                    (B) Sanity-Check Backwards (Only if A is false): If you did not update the end time in the step above, re-evaluate the content immediately before your current end time. Is this segment, previously considered part of the ad, actually host app content?
                        If yes, you must adjust the end time to the correct moment the ad truly ends.
                Step 3.4: Iterate or Finalize
                If you revised the end time in Step 3.3 (either backward or forward), you should repeat the entire process from Step 3.3 with the new, updated end time.
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