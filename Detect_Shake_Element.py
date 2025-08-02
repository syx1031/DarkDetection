import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request, time_to_seconds
from Bbox import Location, Bbox_Description


def detect_shake_element_time_location(client, video, start_time=None, end_time=None):
    class ShakeElement(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which the element (text, icon, et al.) suggesting the user shake their phones appears, should be represented in the format 'mm:ss'.")
        location: Location = Field(...,
                                   description="The location where the element suggesting the user shake their phones appears.")
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

    ShakeElementList = list[ShakeElement]

    # prompt_detect_shake_element = f'''
    #     Task: Did the app display any UI element (text, icon, et al.) in this video from {start_time} to {end_time} suggesting the user shake their phones? List all the timestamp in the format "mm:ss".
    #     Output: You need to describe the location of the element: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon, while width and height represent its size relative to the video’s width and height, respectively.
    # '''

    prompt_detect_shake_element = f'''
    Context: 
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
        2. Definition of "Shake Element": A "Shake Element" is any visual component in a user interface designed to instruct, prompt, or suggest that the user should physically shake their mobile device to trigger an action. A "Shake Element" should meet the criteria for at least one of these three types:
            2.1. Explicit Textual Instructions:
                Description: Any UI element (button, banner, text label) that contains explicit text prompting the user to shake their device.
                Keywords to look for: "Shake", "Shake to...", "摇一摇", "摇动", etc. (in any language).
            2.2. Iconic Representations:
                Description: An icon designed to visually represent the "shake" action.
                Visual Features: Typically depicts a smartphone silhouette, often with motion lines, arrows, or curves around it to signify movement or vibration.
            2.3. Animated Cues:
                Description: A UI element (e.g., a gift box, red envelope, button, card) that uses a shaking or wobbling animation to implicitly suggest a shake interaction.
                Animation Characteristics: The animation should be a small, rapid, and repetitive side-to-side or up-and-down motion.
                
    Task: Your task is to analyze the video and find out all "Shake Element". Please check each UI element referring to the definitions from 2.1 to 2.3 sequentially.
    
    Output: For each "Shake Element" you find, provide the following details. If multiple distinct elements are visible at the same time, list each one separately.
        - `timestamp`: The timestamp "mm:ss" when the element begin to clearly appear.
        - 'location': {Bbox_Description}
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=ShakeElementList,
    )

    content = [video, prompt_detect_shake_element]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
