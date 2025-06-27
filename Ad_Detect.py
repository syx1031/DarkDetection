from google.genai import types
from google import genai
from pydantic import BaseModel, Field, confloat, validator
from typing import Literal
import re
import time
import json
from moviepy import VideoFileClip
import traceback

from utils import upload_file, send_request, get_client, dump_upload_files, time_to_seconds, seconds_to_mmss


class Location(BaseModel):
    x: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="The ratio of the UI element’s left edge to the total video width.")
    y: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="The ratio of the UI element’s lower edge to the total video height.")
    width: confloat(ge=0.0, le=1.0) = Field(...,
                                            description="The ratio of the UI element’s width to the total video width.")
    height: confloat(ge=0.0, le=1.0) = Field(...,
                                             description="The ratio of the UI element’s height to the total video height.")


class PointLocation(BaseModel):
    x: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="The ratio of the UI element’s center x-coordinate to the total video width.")
    y: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="The ratio of the UI element’s center y-coordinate to the total video height.")


PointLocation_Description = ''''x' represents the ratio of the UI element’s center x-coordinate to the total video width, and 'y' represents the ratio of the UI element’s center y-coordinate to the total video height.'''


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
    1. At what time did advertisements appear (use the format xx:xx–xx:xx)? List all time periods during which ads appeared. Note: (1) Although App Store pages or prompt pages requesting users to rate the app can be part of an ad, they are not considered ad interfaces when standalone. (2) The interface for paid ad removal is part of the app’s functional UI and does not count as an ad. (3) Interfaces that merely request the user to watch an ad (e.g., “Watch an ad for...”) are not considered ads.
    2. Reconsider the ad time intervals you identified in Task 1---especially those with similar content or consecutive time intervals---and determine whether they represent different stages of the same ad (e.g., a video followed by an playable demo) or different UI components of the same stage (e.g., a full-screen video and a banner with summary information below it). Then treat these intervals as a single ad for further analysis. List all time periods during which ads appeared again after your reconsideration.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdSegmentList,
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


def detect_purchase_interface(client, video):
    class PurchaseInterface(BaseModel):
        timestamp: str = Field(...,
                            description="The timestamp when the purchase interface appears, should be represented in the format 'mm:ss'.")
        thinking: str = Field(...,
                              description="Explain in detail the reasoning behind your judgment that a purchase interface appeared.")

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

    # 用于整体结构的 list 类型
    PurchaseInterfaceList = list[PurchaseInterface]

    prompt_detect_purchase_interface = '''
    At what time in the video did the app present the user with an interface for purchasing a paid service? Note that a "purchase interface" must include both the amount the user is required to pay and the benefits they will receive. List all the timestamp in the format "mm:ss".
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=PurchaseInterfaceList,
    )

    contents = [video, prompt_detect_purchase_interface]

    # Send request with function declarations
    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=contents,
        config=config,
    )

    # print("Detect Purchase Interfaces:")
    # print(response.text)
    return json.loads(response.text)


def detect_ad_removal_text_time_location(client, video, start_time=None, end_time=None):
    class AdRemovalText(BaseModel):
        timestamp: str = Field(...,
                            description="The timestamp at which the text implying that the user can remove ads appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(..., description="The location where the text implying that the user can remove ads appears.")
        thinking: str = Field(...,
                              description="Explain in detail the reasoning behind your judgment that the text implying ad removal appeared.")

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

    AdRemovalTextList = list[AdRemovalText]

    if start_time and end_time:
        prompt_detect_ad_removal_text = '''
            Did the app display any text in this video that implied or invited the user to remove ads? List all the timestamp in the format "mm:ss". You need to describe the location of the text box: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the text box, while width and height represent its size relative to the video’s width and height, respectively.
            '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=AdRemovalTextList,
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
                types.Part(text=prompt_detect_ad_removal_text)
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


def detect_ad_removal_icon_time_location(client, video, start_time=None, end_time=None):
    class AdRemovalIcon(BaseModel):
        timestamp: str = Field(...,
                            description="The timestamp at which the icon implying that the user can remove ads appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(..., description="The location where the icon implying that the user can remove ads appears.")
        thinking: str = Field(...,
                              description="Explain in detail the reasoning behind your judgment that the icon implying ad removal appeared.")

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

    AdRemovalIconList = list[AdRemovalIcon]

    if start_time and end_time:
        prompt_detect_ad_removal_icon = '''
            Did the app display any icon in this video that implied or invited the user to remove ads? List all the timestamp in the format "mm:ss". You need to describe the location of the icon: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon, while width and height represent its size relative to the video’s width and height, respectively.
            '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=AdRemovalIconList,
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
                types.Part(text=prompt_detect_ad_removal_icon)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Detect Ad Removal Icon:")
        # print(response.text)
        return json.loads(response.text)


def Decide_Paid_Ad_Removal(client, video, purchase_interface, ad_removal_icon, ad_removal_text, start_time=None, end_time=None):
    class PaidAdRemoval(BaseModel):
        timestamp: str = Field(..., description="The timestamp when the purchase interface appears, should be represented in the format 'mm:ss'.")
        paidadremoval: bool = Field(..., description="Whether 'Paid Ad Removal' appears or not.")
        thinking: str = Field(...,description="Provide detailed reasoning about 'Paid Ad Removal'. If it is present, you must explain how the UI elements are combined to form the dark pattern.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    if start_time and end_time:
        prompt_decide_paid_ad_removal = f'''
        Context:
        1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
            a. The persistent red-bordered circle represents the current position of the cursor. 
            b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
            c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
        2. "Paid Ad Removal": Some apps offer a paid option to remove ads.
        
        Auxiliary information:
        1. A purchase interface may have appeared at the following timestamp:
        {purchase_interface}
        2. Texts that implied or invited the user to remove ads have appeared at the following timestamp and location:
        {ad_removal_text}
        3. Icons that implied or invited the user to remove ads have appeared at the following timestamp and location:
        {ad_removal_icon}
        Note: The location of the icon/text is described by x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon/text, while width and height represent its size relative to the video’s width and height, respectively.
        
        Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern “Paid Ad Removal” during {start_time}-{end_time}. Below are common UI element combinations associated with this pattern:
        1. The "purchase interface" displayed "ad removal text" or "ad removal icon", indicating to the user that ads would be removed after making a purchase.
        2. The app displayed "ad removal text" or "ad removal icon", and when the user clicked on it, a "purchase interface" appeared, indicating that a purchase was required to remove ads.
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


def recheck_ads(client, video, start_time, end_time):
    class AdAttribution(BaseModel):
        start_time: str = Field(..., description="The timestamp when the ad starts, should be represented in the format 'mm:ss'. If no ad occurs, set this attribution to '00:00'.")
        end_time: str = Field(..., description="The timestamp when the ad ends, should be represented in the format 'mm:ss'. If no ad occurs, set this attribution to '00:00'.")
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

    # start_time_mmss = seconds_to_mmss(start_time)
    # end_time_mmss = seconds_to_mmss(end_time)
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

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdAttribution,
    )

    content = [video, prompt_recheck_ad]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    # print("Recheck Ad:")
    # print(response.text)
    return json.loads(response.text)


def detect_outside_interface(client, video, start_time=None, end_time=None):
    class Outside_Interface(BaseModel):
        go_outside: bool = Field(..., description="Whether the user gets out of the app or not.")
        outside_interface_type: Literal["Home Screen", "Control Center", "Notification Center", "Browser", "App Switcher", "Others"] = Field(..., description="The interface type, should be in one of the following strings: 'Home Screen', 'Control Center', 'Notification Center', 'Browser', 'App Switcher', 'Others'.")
        go_outside_time: str = Field(..., description="The timestamp when the user gets out of the app, should be represented in the format 'mm:ss'.")
        resume_app_time: str = Field(..., description="The timestamp when the user returns to the app, should be represented in the format 'mm:ss'.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['go_outside_time'] = self.validate_timestamp(data['go_outside_time'])
            data['resume_app_time'] = self.validate_timestamp(data['resume_app_time'])
            super().__init__(**data)

    if end_time and not start_time:
        prompt_detect_outside_interface = f'''
            Context:
            1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
                a. The persistent red-bordered circle represents the current position of the cursor. 
                b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
                c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

            Task: 
            1. Check whether the user resumed the app from an interface out of the app within the two seconds before {end_time}. Note that you only need to examine events that occurred within the two seconds before {end_time}; events after it should be ignored.
            2. If your answer to Task 1 is yes, then identify the type ("Home Screen/Control Center/Notification Center/Browser/App Switcher/Other") of this outside interface, and identify when the user previously left the app and went to the external interface prior to this app resumption.
            '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=Outside_Interface,
        )

        content = [video, prompt_detect_outside_interface]

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Check Outside Interface:")
        # print(response.text)
        return json.loads(response.text)


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

    prompt_decide_app_resumption_ads = f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    2. "App Resumption Ads": When using an app, users may temporarily exit the app by accessing the iPhone’s Control Center or swiping up to return to the Home Screen. Upon returning to the app, they may be forced to view an advertisement before resuming their activity, disrupting their original experience.

    Auxiliary information:
    1. The user may have temporarily left the app and navigated to an external interface during the following time period:
    {outside_interface}
    2. After returning to the app, the user may have been shown an ad immediately during the following time period:
    {recheck_ad}

    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern “App Resumption Ads” in the period {start_time}-{end_time}. Note that only full-screen ads triggered immediately after the user returns to the app qualify as this pattern. Below are common UI element combinations associated with this pattern:
    1. The user temporarily left the app to an "external interface", and returned to the app. Within two seconds of returning to the app, the user was immediately shown a "full-screen ad" without having actively clicked on any ad-related content. Note that this "full-screen ad" must not be displayed before the user left the app.
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


def detect_click_time_location(client, video, start_time=None, end_time=None):
    class Click(BaseModel):
        timestamp: str = Field(...,
                            description="The timestamp at which user click the mouse, accurate to the millsecond and in the format 'mm:ss:xx'.")
        location: PointLocation = Field(..., description="The location where the use clicks.")

        # 自定义验证器确保 period 格式合法
        @classmethod
        def validate_timestamp(cls, value: str) -> str:
            if not re.match(r"^\d{2}:\d{2}:\d{2}$", value):
                raise ValueError("timestamp must be in 'mm:ss:xx' format.")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            # data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    ClickList = list[Click]

    if start_time and end_time:
        prompt_detect_click = '''
        Context:
        1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
            a. The persistent red-bordered circle represents the current position of the cursor. 
            b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        
        Task: Identify the timestamp of each user click, accurate to the millisecond. You need to describe the location of the click: x, y, each ranging from 0 to 1. Here, x and y indicate the relative position of the center of the red circle.
        '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=ClickList,
        )

        start_offset = str(time_to_seconds(start_time)) + 's'
        end_offset = str(time_to_seconds(end_time)) + 's'
        fps = 24

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
                types.Part(text=prompt_detect_click)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Detect Click:")
        # print(response.text)
        return json.loads(response.text)


Hover_Description = ''''start/end_timestamp' represents the timestamp at which user begin/end the click, accurate to the millsecond and in the format 'mm:ss:xx'. 'start/end_location' represents the location where the user begin/end the click.'''


def detect_hover_time_location(client, video, start_time=None, end_time=None):
    class Hover(BaseModel):
        start_timestamp: str = Field(...,
                                     description="The timestamp at which user begin the click, accurate to the millsecond and in the format 'mm:ss:xx'.")
        start_location: PointLocation = Field(..., description="The location where the user begin the click.")
        end_timestamp: str = Field(...,
                                   description="The timestamp at which user complete the click, accurate to the millsecond and in the format 'mm:ss:xx'.")
        end_location: PointLocation = Field(..., description="The location where the user end the click.")

        # 自定义验证器确保 period 格式合法
        @classmethod
        def validate_timestamp(cls, value: str) -> str:
            if not re.match(r"^\d{2}:\d{2}:\d{2}$", value):
                raise ValueError("timestamp must be in 'mm:ss:xx' format.")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            # data['start_timestamp'] = self.validate_timestamp(data['start_timestamp'])
            # data['end_timestamp'] = self.validate_timestamp(data['end_timestamp'])
            super().__init__(**data)

    HoverList = list[Hover]

    if start_time and end_time:
        prompt_detect_hover = '''
        Context:
        1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
            a. The persistent red-bordered circle represents the current position of the cursor. 
            b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        2. Hover: This describes a sequence of actions where the user clicks and holds the button, then drags it across the screen. Visually, this is reflected by a red circle that contracts and darkens in the center while continuously changing position across consecutive frames, until it returns to original state, indicating the end of the click-and-drag action.

        Task: Identify the start and end timestamp and location of each user's hover, accurate to the millisecond.
        '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=HoverList,
        )

        start_offset = str(time_to_seconds(start_time)) + 's'
        end_offset = str(time_to_seconds(end_time)) + 's'
        fps = 24

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
                types.Part(text=prompt_detect_hover)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Detect Hover:")
        # print(response.text)
        return json.loads(response.text)

def detect_watch_ad_text_time_location(client, video, start_time=None, end_time=None):
    class WatchAdText(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which the text implying or suggesting users to watch ads appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(...,
                                   description="The location where the text implying or suggesting users to watch ads appears.")
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

    WatchAdTextList = list[WatchAdText]

    if start_time and end_time:
        prompt_detect_watch_ad_text = '''
            Task: Did the app display any text in this video that implied or invited the user to watch an ad? List all the timestamp in the format "mm:ss". Note that “watch ad text” refers to text indicating to the user clicking on it or a nearby button will lead to watching an ad. It does not refer to text that merely indicates the current content is an ad.
            Output: You need to describe the location of the text box: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the text box, while width and height represent its size relative to the video’s width and height, respectively.
            '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=WatchAdTextList,
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
                types.Part(text=prompt_detect_watch_ad_text)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Detect Watch Ad Text:")
        # print(response.text)
        return json.loads(response.text)


def detect_watch_ad_icon_time_location(client, video, start_time=None, end_time=None):
    class WatchAdIcon(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which the icon implying or suggesting users to watch ads appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(...,
                                   description="The location where the icon implying or suggesting users to watch ads appears.")
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

    WatchAdIconList = list[WatchAdIcon]

    if start_time and end_time:
        prompt_detect_watch_ad_icon = '''
            Task: Did the app display any icon in this video that implied or invited the user to watch an ad? List all the timestamp in the format "mm:ss". Note that “watch ad icon” refers to icon indicating to the user clicking on it or a nearby button will lead to watching an ad. It does not refer to icon that merely indicates the current content is an ad.
            Output: You need to describe the location of the icon: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon, while width and height represent its size relative to the video’s width and height, respectively.
            '''

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            response_mime_type="application/json",
            response_schema=WatchAdIconList,
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
                types.Part(text=prompt_detect_watch_ad_icon)
            ]
        )

        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            contents=content,
            config=config,
        )

        # print("Detect Watch Ad Icon:")
        # print(response.text)
        return json.loads(response.text)


def Decide_Unexpected_Full_Screen_Ads(client, video, recheck_ads_time, click_time_location, hover_time_location, watch_ad_icon_time_location, watch_ad_text_time_location, start_time, end_time):
    class UnexpectedFullScreenAds(BaseModel):
        unexpected_full_screen_ads: bool = Field(..., description="Whether 'Unexpected Full-Screen Ads' appears or not. ")
        click_or_hover_time: str = Field(...,
                               description="The timestamp when the user clicks or hovers on the button which triggers the full-screen ad, must in format 'mm:ss'. If there is no full-screen ad, or if the ad was not triggered by a user action, set this attribute to '00:00'.")
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
            data['click_or_hover_time'] = self.validate_timestamp(data['click_or_hover_time'])
            data['ad_start_time'] = self.validate_timestamp(data['ad_start_time'])
            data['ad_end_time'] = self.validate_timestamp(data['ad_end_time'])
            super().__init__(**data)

    prompt_decide_unexpected_full_screen_ads = f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    2. Hover: This describes a sequence of actions where the user clicks and holds the button, then drags it to another place on the screen. Visually, this is reflected by a red circle that contracts and darkens in the center while continuously changing position across consecutive frames, until it returns to original state, indicating the end of the click-and-drag (hover) action.
    3. Unexpected Full-Screen Ads: These ads may manifest in two distinct forms: either triggered by user interaction with a button (denoted as “Button-Triggered Unexpected Ads”), or appearing spontaneously without any user gesture (denoted as “Unprompted Intrusive Ads”).

    Auxiliary information:
    1. The user identified a complete advertisement during the following time period:
    {recheck_ads_time}
    2. In the few seconds leading up to this ad, the user clicked at the following timestamps and screen positions. Note that {PointLocation_Description}
    {click_time_location}
    3. In the few seconds leading up to this ad, the user hovered at the following timestamps and screen positions. Note that {Hover_Description}
    {hover_time_location}
    4. A few seconds before the advertisement appeared, the app displayed the following text implying or suggesting that users should watch ads:
    {watch_ad_text_time_location}
    5. A few seconds before the advertisement appeared, the app displayed the following icon implying or suggesting that users should watch ads:
    {watch_ad_icon_time_location}

    Task: Based on the auxiliary information, analyze whether these UI elements in auxiliary information constitute the dark pattern "Unexpected Full-Screen Ads" in the period {start_time}-{end_time}. Below are common UI element combinations associated with this pattern:
    1. Few seconds before the “full-screen ad” appeared, the user performed a “click” or “hover” action, but these actions did not land on or pass over any “watch ad text” or “watch ad icon.” Nevertheless, the “full-screen ad” appeared afterward.
    2. Few seconds before the “full-screen ad” appeared, the user didn't perform any “click” or “hover” action. Nevertheless, the "full-screen ad" appeared afterward.
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


def run_detect(client, video, local_path):
    result_dict = {"prediction": {}}

    video_local_info = VideoFileClip(local_path)
    video_duration = int(video_local_info.duration)
    # fps = round(video.fps)

    # How about "Barter for Ad-Free Privilege"?

    result_dict["prediction"]["Paid Ad Removal"] = {"video-level": False, 'instance-level': []}

    purchase_interface_time = detect_purchase_interface(client, video)
    result_dict["Purchase Interface"] = {'Result': purchase_interface_time}
    for purchase_interface in purchase_interface_time:
        timestamp = purchase_interface["timestamp"]
        start_time = seconds_to_mmss(max(0, time_to_seconds(timestamp) - 7))
        end_time = seconds_to_mmss(min(video_duration, time_to_seconds(timestamp) + 7))
        ad_removal_icon_time_location = detect_ad_removal_icon_time_location(client, video, start_time, end_time)
        ad_removal_text_time_location = detect_ad_removal_text_time_location(client, video, start_time, end_time)
        # Detect "Paid Ad Removal"
        Paid_Ad_Removal = Decide_Paid_Ad_Removal(client, video, purchase_interface, ad_removal_icon_time_location, ad_removal_text_time_location, start_time, end_time)
        result_dict["Purchase Interface"][timestamp] = {
            'Ad Removal Icon': {'Parameter': [start_time, end_time], 'Result': ad_removal_icon_time_location},
            'Ad Removal Text': {'Parameter': [start_time, end_time], 'Result': ad_removal_text_time_location},
            'Paid Ad Removal': {'Parameter': [start_time, end_time], 'Result': Paid_Ad_Removal},
        }

        if Paid_Ad_Removal["paidadremoval"]:
            result_dict["prediction"]["Paid Ad Removal"]["video-level"] = True
            result_dict["prediction"]["Paid Ad Removal"]["instance-level"].append(Paid_Ad_Removal["timestamp"])

    # reward_icon_time_location = detect_reward_icon(client, video)
    # reward_text_time_location = detect_reward_text(client, video)
    # for reward in reward_icon_time_location + reward_text_time_location:
    #     timestamp = reward["timestamp"]
    #     watch_ad_text_time_location = detect_watch_ad_text(client, video, timestamp)
    #     watch_ad_icon_time_location = detect_watch_ad_icon(client, video, timestamp)
    #     # Detect "Reward-Based Ads"

    result_dict["prediction"]["App Resumption Ads"] = {"video-level": False, 'instance-level': []}
    result_dict["prediction"]["Unexpected Full-Screen Ads"] = {"video-level": False, 'instance-level': []}

    ads_time = detect_ads(client, video)
    result_dict['Ad'] = {'Result': ads_time}

    result_dict['Ad']['Further Check'] = {}
    for ad in ads_time:
        result_timestamp = start_time = ad["start_timestamp"]
        end_time = ad["end_timestamp"]
        recheck_ads_time = recheck_ads(client, video, start_time, end_time)
        result_dict["Ad"]['Further Check'][result_timestamp] = {
            'Recheck Ad': {'Parameter': [start_time, end_time], 'Result': recheck_ads_time},
        }
        if bool(recheck_ads_time["full_screen"]):
            start_time = recheck_ads_time["start_time"]
            end_time = recheck_ads_time["end_time"]
            outside_interface_time = detect_outside_interface(client, video, end_time=start_time)
            result_dict["Ad"]['Further Check'][result_timestamp].update({
                'Outside Interface': {'Parameter': ['end_time=' + start_time], 'Result': outside_interface_time}
            })
            if bool(outside_interface_time["go_outside"]) and time_to_seconds(outside_interface_time["go_outside_time"]) > 0:
                start_time = outside_interface_time["go_outside_time"]
                # Detect "App Resumption Ads"
                App_Resumption_Ads = Decide_App_Resumption_Ads(client, video, recheck_ads_time, outside_interface_time, start_time, end_time)
                result_dict["Ad"]['Further Check'][result_timestamp].update({
                    'App Resumption Ads': {'Parameter': [start_time, end_time], 'Result': App_Resumption_Ads}
                })

                if App_Resumption_Ads["app_resumption_ads"]:
                    result_dict["prediction"]["App Resumption Ads"]["video-level"] = True
                    result_dict["prediction"]["App Resumption Ads"]["instance-level"].append(App_Resumption_Ads["ad_start_time"])

        start_time = seconds_to_mmss(max(0, time_to_seconds(recheck_ads_time["start_time"])-3))
        end_time = recheck_ads_time["end_time"]
        if bool(recheck_ads_time["full_screen"]) and time_to_seconds(start_time) > 0:
            click_time_location = detect_click_time_location(client, video, start_time, recheck_ads_time["start_time"])
            hover_time_location = detect_hover_time_location(client, video, start_time, recheck_ads_time["start_time"])
            watch_ad_icon_time_location = detect_watch_ad_icon_time_location(client, video, start_time, recheck_ads_time["start_time"])
            watch_ad_text_time_location = detect_watch_ad_text_time_location(client, video, start_time, recheck_ads_time["start_time"])
            # Detect "Unexpected Full-Screen Ads"
            Unexpected_Full_Screen_Ads = Decide_Unexpected_Full_Screen_Ads(client, video, recheck_ads_time, click_time_location, hover_time_location, watch_ad_icon_time_location, watch_ad_text_time_location, start_time, end_time)
            result_dict["Ad"]['Further Check'][result_timestamp].update({
                'Click before Ad': {'Parameter': [start_time, recheck_ads_time["start_time"]], 'Result': click_time_location},
                'Hover before Ad': {'Parameter': [start_time, recheck_ads_time["start_time"]], 'Result': hover_time_location},
                'Watch Ad Icon': {'Parameter': [start_time, recheck_ads_time["start_time"]], 'Result': watch_ad_icon_time_location},
                'Watch Ad Text': {'Parameter': [start_time, recheck_ads_time["start_time"]], 'Result': watch_ad_text_time_location},
                'Unexpected Full-Screen Ads': {'Parameter': [start_time, end_time], 'Result': Unexpected_Full_Screen_Ads},
            })

            if Unexpected_Full_Screen_Ads["unexpected_full_screen_ads"]:
                result_dict["prediction"]["Unexpected Full-Screen Ads"]["video-level"] = True
                result_dict["prediction"]["Unexpected Full-Screen Ads"]["instance-level"].append(
                    Unexpected_Full_Screen_Ads["ad_start_time"])

    return result_dict


def upload_file_and_run_detect(video_local_path):
    client = get_client(local_path=video_local_path)

    try:
        video_file = upload_file(client, video_local_path)
    except Exception as e:
        return {'broken_file_upload': True, 'error_information': traceback.format_exc()}
    # video_file = upload_file(client, video_local_path)
    dump_upload_files()

    try:
        return run_detect(client, video_file, video_local_path)
    except Exception as e:
        return {'broken_client': True, 'error_information': traceback.format_exc()}
    # return run_detect(client, video_file, video_local_path)


if __name__ == "__main__":
    client = get_client(key='AIzaSyD22lWTZq6SPGgeQicUYic8G1oukNnntxo')

    video_local_path = "E:\\DarkDetection\\dataset\\syx\\us\\470909765-尚宇轩.mp4"

    cloud_name = "files/9e58bj25wkus"
    video_file = client.files.get(name=cloud_name)

    # video_file = client.files.upload(file=video_local_path)
    # while not video_file.state or video_file.state.name != "ACTIVE":
    #     # print("Processing video...")
    #     # print("File state:", video_file.state)
    #     time.sleep(5)
    #     video_file = client.files.get(name=video_file.name)
    #
    # print(video_file.name)

    run_detect(client, video_file, video_local_path)
