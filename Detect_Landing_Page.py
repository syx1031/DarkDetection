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

    prompt_detect_landing_page = f'''
        Context: 
        1. Landing Page: A landing page refers to the webpage that the user ultimately reaches after clicking on an ad. It is a standalone page separate from the ad, typically associated with a specific advertising campaign or marketing goal, and is designed to prompt the user to take an action such as registering, purchasing, or downloading. While the content of the landing page aligns with the ad, its UI design often differs significantly and is presented through third-party interfaces, such as an App Store download page or a website within the browser.
        
        Task: Did the ad during {start_time}-{end_time} displays any landing page? List the timestamp it occurs in the format "mm:ss".
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
