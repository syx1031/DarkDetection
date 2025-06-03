from google.genai import types
from google import genai
from typing import Union
import os
import time

# Define a function that the model can call to control smart lights
Detect_Ad_Close_Button = {
    "name": "Detect_Ad_Close_Button",
    "description": "Provide the name of the user-uploaded video to this function, along with the frame number you want to analyze (indicated by the yellow number at the bottom-right corner of each frame). The function will return whether an ad close button appears in that frame. If it does, it will also return the coordinates of the bottom-left corner of the close button in the format (x, y).",
    "parameters": {
      "type": "object",
      "properties": {
        "Frame_Num": {
          "type": "integer"
        },
        "Video_Name": {
          "type": "string"
        }
      },
      "required": [
        "Frame_Num",
        "Video_Name"
      ]
    },
}


# This is the actual function that would be called based on the model's suggestion
def Detect_Ad_Close_Button(Frame_Num: int, Video_Name: str) -> dict[str, Union[bool, tuple[int, int]]]:
    return {"Has_Close_Button": True}

prompt = '''
Context:
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse. 
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

2. The following are some deceptive or malicious UI designs related to advertisements found in apps or ads, which are referred to as "ads dark patterns":
A. "App Resumption Ads": When using an app, users may temporarily exit the app by accessing the iPhone’s Control Center or swiping up to return to the Home Screen.
B. "Unexpected Full-Screen Ads": These ads may manifest in two distinct forms: either triggered by user interaction with a button (denoted as “Button-Triggered Unexpected Ads”), or appearing spontaneously without any user input (denoted as “Unprompted Intrusive Ads”). Note that (1) only ads that appear during normal app usage (excluding app launching or returning to the app from the background or Home Screen) may exhibit this dark pattern. (2) An landing page (e.g. app information in App Store or a website) triggered by an ad should not be considered a separate advertisement and therefore should not be used as evidence for this dark pattern. (3) The interface for paid ad removal is part of the app’s functional UI and does not count as this type of dark pattern.
C. "Auto-Redirect Ads": An Auto-Redirect Ad automatically redirects to its landing page after the ad concludes, without the user clicking a “Skip” or “Close” button.
D. "Long Ad/Many Ads":  Excessive exposure to ads—whether due to an overabundance of ads or excessively long ad durations—negatively impacts the user experience.
E. "Barter for Ad-Free Privilege": Some apps offer users the option to remove ads (or access an ad-free experience) through methods such as watching videos, viewing ads.
F. "Paid Ad Removal": Some apps offer a paid option to remove ads.
G. "Reward-Based Ads": Users may be required to watch ads in exchange for other benefits, such as “earning game items” or “unlocking additional features”.
H. "Ad Without Exit Options":  Some ads do not provide a close button or delay their appearance to force users to watch them.
I. "Ad Closure Failure": After clicking the close button, an ad may fail to close as expected. There are several more detailed manifestations:
    a. "Multi-Step Ad Closure": requires users to complete multiple dismissal actions, as the initial close button click merely redirects to another advertisement page.
    b. "Closure Redirect Ads":  immediately direct users to promotional landing pages, typically app store destinations, upon closure attempts.
    c. "Forced Ad-Free Purchase Prompts": present subscription offers immediately after ad dismissal, effectively transforming the closure action into a monetization opportunity. 
J. "Increased Ads with Use": Initially, an app may not display startup ads or show very few ads. However, after repeated use, users may begin to see more ads.
K. "Gesture-Induced Ad Redirection": Some ads are triggered by easily detected user actions (e.g., shaking the phone or hovering a finger over the ad), which can result in inadvertent activation.
    a. “Shake-to-open”: ads are triggered by movement sensors. Note that text or icons in the ad that prompt the user to shake the phone can serve as sufficient evidence that the ad uses this mechanism.
    b. “Hover-to-open”: ads are activated when a finger hovers over the ad. Developers may set low action thresholds to capture slight user movements.
L. "Button-Covering Ads": Ads may obscure system or functional buttons (e.g., Home Indicator) within the app, preventing users from interacting with them.
M. "Multiple Close Buttons": Some ads display multiple close buttons simultaneously, making it difficult for users to choose the correct one.
N. "Bias-Driven UI Ads": In ads where users are presented with options, the visual design often emphasizes choices that benefit the advertiser, such as directing users to a landing page or prompting them to watch another ad. 
O. "Disguised Ads": Ads may be designed to closely resemble regular content within apps or the system UI, making them difficult for users to recognize as ads.


Task: First, watch the video and analyze the questions below, then determine whether each of the 15 ad dark patterns appeared in the video.
1. At what time did advertisements appear (use the format xx:xx–xx:xx)? List all time periods during which ads appeared. Note: Standalone App Store pages or prompts requesting users to rate the app are not considered ad interfaces themselves, although they may be related to "Auto-redirect Ads" or "Ad Closure Failure."
2. Reconsider the ad time intervals you identified---especially those with similar content or consecutive time intervals---and determine whether they represent different stages of the same ad (e.g., a video followed by an playable demo) or different UI components of the same stage (e.g., a full-screen video and a banner with summary information below it). Then treat these intervals as a single ad for analysis.
3. Analyze the time intervals you listed one by one, focusing on the following questions:
    3.1. (Corresponding to "Ad Without Exit Options") Determine whether the advertisement provides a close button after the first 2 seconds or never provides one. (In addition to the common “X”, any icon indicating skip, fast-forward, etc., should also be considered a close button.)
    3.2. (Corresponding to "Multiple Close Buttons") Did the ad simultaneously provide multiple close buttons?
    3.3. (Corresponding to "Bias-Driven UI Ads") Check whether there is a pair of buttons placed next to each other, where the UI design uses strong visual contrast to highlight the option that benefits the advertiser. Note that such a pair of buttons must be adjacent in spatial position and both of them should belong to an ad.
    3.4. (Corresponding to "Button-Covering Ads") Check whether the ad overlaps with in-app buttons or system buttons (such as the Home Indicator). Note: You should only identify this dark pattern if you are certain that the obscured button is a legitimate app or system function, and you observe the user attempting to use the button but failing.


Output: List all the ad dark patterns appeared and your reasoning behind the judgment, one per line. If a dark pattern's instances appear multiple times, you only need to list it once.
'''

# Configure the client and tools
client = genai.Client(api_key="AIzaSyD6ClWdvvtGbm600-BvopMy4vzkEkqkedI")
# tools = types.Tool(function_declarations=[Detect_Ad_Close_Button])
# config = types.GenerateContentConfig(tools=[tools])
config = types.GenerateContentConfig()
video_file = client.files.upload(file="Unprompted_Intrusive_Ads_(Unexpected_Full-Screen_Ads)_en_1425445169.mp4")

# Poll until the video file is completely processed (state becomes ACTIVE).
while not video_file.state or video_file.state.name != "ACTIVE":
    # print("Processing video...")
    # print("File state:", video_file.state)
    time.sleep(5)
    video_file = client.files.get(name=video_file.name)

# Send request with function declarations
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-04-17",
    contents=[video_file, prompt],
    config=config,
)

# Check for a function call
if response.candidates[0].content.parts[0].function_call:
    function_call = response.candidates[0].content.parts[0].function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")
    #  In a real app, you would call your function here:
    #  result = schedule_meeting(**function_call.args)
else:
    print("No function call found in the response.")
    print(response.text)

