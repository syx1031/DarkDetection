import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Bbox import Location


def detect_landing_page_time(client, video, start_time=None, end_time=None):
    class LandingPage(BaseModel):
        landing_page: bool = Field(..., description="Decide whether a 'landing page' occurs or not.")
        timestamp: str = Field(...,
                               description="The timestamp at which the landing page appears in the ad. Should be represented in the format 'mm:ss'. If no landing page occurs, please fill '00:00'.")
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

    # LandingPageList = list[LandingPage]

    # prompt_detect_landing_page = f'''
    #     Context:
    #     1. Landing Page: A landing page refers to the webpage that the user ultimately reaches after clicking on an ad. It is a standalone page separate from the ad, typically associated with a specific advertising campaign or marketing goal, and is designed to prompt the user to take an action such as registering, purchasing, or downloading. While the content of the landing page aligns with the ad, its UI design often differs significantly and is presented through third-party interfaces, such as an App Store download page or a website within the browser.
    #
    #     Task: Did the ad during {start_time}-{end_time} displays any landing page? List the timestamp it occurs in the format "mm:ss".
    # '''

    prompt_detect_landing_page = f'''
    Context: 
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
        2. Definition of "Ad Landing Page": A Landing Page is defined as the specific, interactive destination a user arrives from an in-app ad. It is not part of the ad's video creative itself but is presented within the recognizable UI frame like a third-party application. 
            2.1 To be classified as a Landing Page, the screen must show a Third-Party UI Frame. This is the most critical evidence. The page is displayed inside a larger application context, such as:
                - A web browser (e.g., Chrome, Safari), indicated by a visible URL bar, tabs, or navigation buttons.
                - An official app store (e.g., Apple App Store, Google Play Store), indicated by their unique headers, search bars, "Get"/"Install" buttons in their standard positions, and overall layout.
            2.2 What is NOT a Landing Page (Common mistakes to avoid): 
                - Ad End Cards or Summary Screens: Do not classify final screens of the ad as landing pages, even if they show the app's name, logo, features, and a "Download Now" button. If these screens are full-screen and lack a browser or app store frame, they are just part of the ad's creative.
                - In-Game/In-App Interfaces: Screens showing the gameplay, menus, or tutorials of the advertised product are not landing pages.
                - Simple Text Overlay: A URL string (e.g., "www.example.com") shown as text on the screen is not a landing page.
            
    Task: Your task is to analyze the video within the period {start_time} to {end_time}, and find out the "Landing Page". Please check each UI element referring to the definitions from 2.1 to 2.2 sequentially.
    
    Output: For each "Landing Page" you find, provide the following details.
        - `timestamp`: The timestamp "mm:ss" when the element begin to clearly appear.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=LandingPage,
    )

    content = [video, prompt_detect_landing_page]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
