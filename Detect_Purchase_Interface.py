import re
import json
from pydantic import BaseModel, Field
from google.genai import types

from utils import send_request


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
            # data['timestamp'] = self.validate_timestamp(data['timestamp'])
            super().__init__(**data)

    # 用于整体结构的 list 类型
    PurchaseInterfaceList = list[PurchaseInterface]

    # prompt_detect_purchase_interface = '''
    # Context:
    #     1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    #         a. The persistent red-bordered circle represents the current position of the cursor.
    #         b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    #         c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
    #
    # Task:
    # 1. At what time in the video did the app present the user with an interface for purchasing a paid service? Note that a "purchase interface" must include both the amount the user is required to pay and the benefits they will receive. List all the timestamp in the format "mm:ss".
    # '''

    prompt_detect_purchase_interface = f'''
    Context: 
        1. You will be analyzing a video segment based on the following information:
            a. Video Source: The video is a screen recording of a user interacting with an iPhone app, with a mouse connected.
            b. Cursor Representation: A persistent red-bordered circle on the screen represents the mouse cursor's position.
            c. Click Indication: A click is indicated when the cursor translates from a red circle to a yellow square.
        2. Definition of "Purchase Interface": A "Purchase Interface" is a screen, pop-up, or dedicated section within the app specifically designed to solicit a payment from the user for a service, subscription, or premium feature. The interface may display the following elements:
            2.1. Explicit Pricing: It must clearly state the cost of the service (e.g., "$4.99/month", "¥50/year", "Unlock for $9.99"). This is an element that a "Purchase Interface" must display.
            2.2. List of Benefits/Features: It may detail what the user will receive in exchange for payment (e.g., "Ad-free experience," "Access to all filters," "Unlimited cloud storage"). This is often presented as a list with checkmarks (✓).
            2.3. A Clear Call-to-Action (CTA): It may feature a prominent, clickable button that initiates the transaction or a trial period leading to a transaction. Examples include "Subscribe Now," "Upgrade to Pro," "Start Free Trial," or "Continue."
            2.4 What does NOT qualify:
                - A simple button labeled "Go Pro" or "Upgrade" in a settings menu is NOT a purchase interface unless the price and benefits are displayed after the user's click on the button.
                - The native iOS payment confirmation pop-up (the one asking for Apple ID password, Face ID, or Touch ID) that appears after the user has already initiated a purchase is NOT the target interface. We are looking for the screen within the app that prompts this action.
            
    Task: Your task is to analyze the video and find out all "Purchase Interface". Please check each UI element referring to the definitions from 2.1 to 2.4 sequentially.
        
    Output: For each "Purchase Interface" you find, provide the following details. If multiple distinct elements are visible at the same time, list each one separately.
        - `timestamp`: The timestamp "mm:ss" when the element begin to clearly appear.
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
