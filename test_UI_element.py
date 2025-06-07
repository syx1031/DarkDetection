from google import genai
from google.genai import types
import time


prompt_unexpected_full_screen_ads = '''
Context:
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
2. "Unexpected Full-Screen Ads": These ads may manifest in two distinct forms: either triggered by user interaction with a button (denoted as “Button-Triggered Unexpected Ads”), or appearing spontaneously without any user input (denoted as “Unprompted Intrusive Ads”). Note that (1) only ads that appear during normal app usage (excluding app launching or returning to the app from the background or Home Screen) may exhibit this dark pattern. (2) An landing page (e.g. app information in App Store or a website) triggered by an ad should not be considered a separate advertisement and therefore should not be used as evidence for this dark pattern. (3) The interface for paid ad removal is part of the app’s functional UI and does not count as this type of dark pattern.
Task: 
1. “Watch Ad" Icon: 你认为视频中是否出现过暗示用户”点击当前按钮会导致观看一个广告“的icon？如果有，出现的时间点和位置是什么？
2. "Watch Ad" Text: 你认为视频中是否出现过暗示用户”点击当前按钮会导致观看一个广告“的文本？如果有，出现的时间点和位置是什么？
3. Full-Screen Ads: 你认为视频中是否出现过全屏或占据大部分屏幕的广告？如果有，出现的时间点是什么？
4. Click: 你认为视频中用户是否进行过鼠标点击？注意鼠标点击反映在屏幕上就是红圈收缩且中心变暗。如果有，出现的时间点和点击位置是什么？
5. Hover: 你认为视频中用户是否摁住并拖拽过鼠标？注意这种拖拽反映在屏幕上就是红圈收缩且中心变暗，随后保持这个状态在屏幕上移动。如果有，出现的时间点和开始的位置是什么？
6. Ad Dark Pattern "Unexpected Full-Screen Ads"是否出现在视频中？
'''

prompt_barter = '''
Context:
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
2. "Barter for Ad-Free Privilege": Some apps offer users the option to remove ads (or access an ad-free experience) through methods such as watching videos, viewing ads.
Task: 
1. "Watch Ad" Icon: 你认为视频中是否出现过暗示用户”点击当前按钮会导致观看一个广告“的icon？如果有，出现的时间点和位置是什么？
2. "Watch Ad" Text: 你认为视频中是否出现过暗示用户”点击当前按钮会导致观看一个广告“的文本？如果有，出现的时间点和位置是什么？
3. "Ad Removal" Icon: 你认为视频中是否出现过表达“点击当前按钮或执行某种方法来暂时或永久移除app内广告”含义的icon？如果有，出现的时间点和位置是什么？
4. “Ad Removal” Text: 你认为视频中是否出现过表达“点击当前按钮或执行某种方法来暂时或永久移除app内广告”含义的文本？如果有，出现的时间点和位置是什么？
5. Ad Dark Pattern "Barter for Ad-Free Privilege"是否出现在视频中？
'''

prompt_paid = '''
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
2. "Paid Ad Removal": Some apps offer a paid option to remove ads.

Task:
1. "Purchase Interface": 你认为视频中是否出现过提供给用户付费购买app内服务机会的界面？如果有，出现的时间点和位置是什么？
2. "Ad Removal" Icon: 你认为视频中是否出现过表达“点击当前按钮或执行某种方法来暂时或永久移除app内广告”含义的icon？如果有，出现的时间点和位置是什么？
3. “Ad Removal” Text: 你认为视频中是否出现过表达“点击当前按钮或执行某种方法来暂时或永久移除app内广告”含义的文本？如果有，出现的时间点和位置是什么？
4. Ad Dark Pattern "Paid Ad Removal"是否出现在视频中？
'''

prompt_reward = '''
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
2. "Reward-Based Ads": Users may be required to watch ads in exchange for other benefits, such as “earning game items” or “unlocking additional features”.

Task:
1. "Watch Ad" Icon: 你认为视频中是否出现过暗示用户”点击当前按钮会导致观看一个广告“的icon？如果有，出现的时间点和位置是什么？
2. “Watch Ad” Text: 你认为视频中是否出现过暗示用户”点击当前按钮会导致观看一个广告“的文本？如果有，出现的时间点和位置是什么？
3. "Reward" Icon: 你认为视频中是否出现过提示用户“点击能获得奖励”的icon？例如像是钱袋、金币等icon。如果有，出现的时间点和位置是什么？
4. "Reward" Icon: 你认为视频中是否出现过提示用户“点击能获得奖励”的文本？例如像是“解锁特性、获得奖励、2X金币”等。如果有，出现的时间点和位置是什么？
5. Ad Dark Pattern "Reward-Based Ads"是否出现在视频中？
'''

prompt_increased = '''
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

Task:
1. Ad: 你认为视频中是否出现过任何尺寸的广告？如果有，出现的时间点和位置是什么？
2. Home Screen: 你认为视频中是否出现过手机的Home Screen界面？如果有，出现的时间点是什么？
3. Close app: 你认为视频中用户是否彻底关闭过app？如果有，出现的时间点是什么？
4. Reopen app: 你认为视频中用户是否重新打开过app？如果有，出现的时间点是什么？
'''

client = genai.Client(api_key='AIzaSyD22lWTZq6SPGgeQicUYic8G1oukNnntxo')

video_local_path = "E:\\DarkDetection\\Exemplars_for_UI_Element_test\\Increased Ads with Use\\1166891071.mp4"
video_file = client.files.upload(file=video_local_path)
while not video_file.state or video_file.state.name != "ACTIVE":
    print("Processing video...")
    print("File state:", video_file.state)
    time.sleep(5)
    video_file = client.files.get(name=video_file.name)

config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,
        thinking_budget=24576,  # max thinking
    ),
)
content = [video_file, prompt_increased]

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    # model="gemini-2.0-flash",
    contents=content,
    config=config,
)

print(response.text)
