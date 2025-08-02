import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request
from Bbox import Location, Bbox_Description


def detect_reward_element_time_location(client, video, start_time=None, end_time=None):
    class RewardElement(BaseModel):
        timestamp: str = Field(...,
                               description="The timestamp at which the element (text, icon, et al.) displaying rewards to the user, such as unlocking more app features or offering in-game currency or items. Should be represented in the format 'mm:ss'.")
        location: Location = Field(...,
                                   description="The location where the element displaying rewards appears.")
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

    RewardElementList = list[RewardElement]

    # prompt_detect_reward_element = '''
    #     Task: Did the app display any UI element (text, icon, et al.) in this video that displays rewards to the user, such as unlocking more app features or offering in-game currency or items? List all the timestamp in the format "mm:ss". Note that "rewards" may include in-game currency, items, doubling game earnings, or unlocking new features in the app or game.
    #     Output: You need to describe the location of the element: x, y, width, and height, each ranging from 0 to 1. Here, x and y indicate the relative position of the lower-left corner of the icon, while width and height represent its size relative to the video’s width and height, respectively.
    # '''

    prompt_detect_reward_element = f'''
    Context: 
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
        2. Definition of "Reward Element": A Reward Element is any visual component (e.g., text, icon, button, pop-up, animation) that explicitly offers, grants, or confirms a reward to the user. 
            2.1 "Reward Element" includes the following categories:
                - Ad-based Incentives: Buttons/prompts to watch an ad for a benefit (e.g., "Watch Ad to Double Coins," "Get a free life").
                - Claimable Rewards: Buttons or prompts for daily logins, mission completions, or level-up bonuses (e.g., "Claim Daily Bonus," "Collect Reward").
                - Chance-based Rewards: UI for loot boxes, prize wheels, or gachas that can be activated for free or with in-game currency (e.g., "Free Spin," "Open Chest").
            2.2 Exclusions: What is NOT a "Reward Element"?
                - Static Balance Displays: The persistent currency counter (e.g., the coin total at the top of the screen) is a balance indicator, not a reward event. However, an animation of coins flying into it should be labeled.
                - Standard Navigation: Buttons like "Continue," "Next Level," or "Retry" that do not offer any explicit bonus.
                - Real-Money Purchases: Any UI element prompting the user to spend real-world currency (e.g., "$0.99," "Buy Now").
                - Standard Score Screens: Game-over or level-complete screens that only show performance metrics (score, time) without highlighting any bonus items or currency earned.
                
    Task: Your task is to analyze the video and find out all "Reward Elements". Please check each UI element referring to the definitions from 2.1 to 2.2 sequentially.
        
    Output: For each "Reward Element" you find, provide the following details. If multiple distinct elements are visible at the same time, list each one separately.
        - `timestamp`: The timestamp "mm:ss" when the element begin to clearly appear.
        - 'location': {Bbox_Description}
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=RewardElementList,
    )

    content = [video, prompt_detect_reward_element]

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=content,
        config=config,
    )

    return json.loads(response.text)
