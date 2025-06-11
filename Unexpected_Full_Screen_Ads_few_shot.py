from google import genai
from google.genai import types
import time

client = genai.Client(api_key='AIzaSyD22lWTZq6SPGgeQicUYic8G1oukNnntxo')

video_local_path = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Unexpected Full-Screen Ads\\1539412097.mp4"
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

prompt_introduce_context = '''
Context:
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
2. The definition of "Unexpected Full-Screen Ads": These ads may manifest in two distinct forms: either triggered by user interaction with a button (denoted as “Button-Triggered Unexpected Ads”), or appearing spontaneously without any user input (denoted as “Unprompted Intrusive Ads”). Note that (1) only ads that appear during normal app usage (excluding app launching or returning to the app from the background or Home Screen) may exhibit this dark pattern. (2) An landing page (e.g. app information in App Store or a website) triggered by an ad should not be considered a separate advertisement and therefore should not be used as evidence for this dark pattern. (3) The interface for paid ad removal is part of the app’s functional UI and does not count as an ad.
'''
prompt_exemplars = '''
Exemplars: There are several exemplars related to "Unexpected Full-Screen Ads", each followed by a detailed analysis:
'''
video_shot_1 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Unexpected Full-Screen Ads\\6447461923.mp4"
video_bytes_1 = open(video_shot_1, 'rb').read()
video_introduce_1 = '''
Exemplar 1. This is an example where Unexpected Full-Screen Ads did not occur. Although at 0:01 the user tapped the GET IT button and was shown a full-screen ad, a closely positioned icon to the left of the GET IT button (resembling a video play symbol) already signaled that tapping the button would lead to an ad, thereby setting appropriate user expectations.
'''
video_shot_2 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Unexpected Full-Screen Ads\\6450923985.mp4"
video_bytes_2 = open(video_shot_2, 'rb').read()
video_introduce_2 = '''
Exemplar 2. This is an example where Unexpected Full-Screen Ads did not occur. At 0:03, the user tapped the SKIP button in the upper-left corner, which triggered an ad to appear. Positioned above the SKIP button was an icon resembling a video player, conveying the message that the user would need to "watch an ad to skip." Therefore, the appearance of the ad following this interaction was expected.
'''
video_shot_3 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Unexpected Full-Screen Ads\\6475673897.mp4"
video_bytes_3 = open(video_shot_3, 'rb').read()
video_introduce_3 = '''
Exemplar 3. This is an example where Unexpected Full-Screen Ads occurred. At 0:01, the user completed a round of the game and received the feedback "WELL DONE." Without tapping any buttons on the screen, the app proactively displayed a full-screen ad at 0:03. This unsolicited interruption constitutes an instance of Unexpected Full-Screen Ads.
'''
video_shot_4 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Unexpected Full-Screen Ads\\764595159.mp4"
video_bytes_4 = open(video_shot_4, 'rb').read()
video_introduce_4 = '''
Exemplar 4: This is an example where Unexpected Full-Screen Ads occurred. At 0:01, the user tapped the button in the lower-left corner, which contained no indication that it was related to advertising. Nevertheless, the user was subsequently shown a full-screen ad.
'''
prompt_task = '''
3. Task: Complete following tasks and check if "Unexpected Full-Screen Ads" has occurred in the video provided by the user:
    Q1: Among the ads in the video, which ones were displayed in full-screen? Note: Ads that initially appear in a non-full-screen format and only expand to full-screen or open a landing page after user interaction should not be counted.
    Q2: Among the ads you identified in Q1, which ones were triggered by the user clicking a normal functional button within the app? Note: "Normal functional buttons" do not include buttons that have indicated “watch ad” through text or icons.
    Q3: Among the ads you identified in Q1, which ones appeared without any gesture input from the user?
    Q4: Discuss the different between Exemplar 1, 2, 3, 4, and user's video.
    Decision: If any ad satisfies both Q1 and Q2, or both Q1 and Q3, then you should determine that "Unexpected Full-Screen Ads" is present.
'''
prompt_task_without_exemplars = '''
3. Task: Complete following tasks and check if "Unexpected Full-Screen Ads" has occurred in the video provided by the user:
    Q1: Among the ads in the video, which ones were displayed in full-screen? Note: Ads that initially appear in a non-full-screen format and only expand to full-screen or open a landing page after user interaction should not be counted.
    Q2: Among the ads you identified in Q1, which ones were triggered by the user clicking a normal functional button within the app? Note: "Normal functional buttons" do not include buttons that have indicated “watch ad” through text or icons.
    Q3: Among the ads you identified in Q1, which ones appeared without any gesture input from the user?
    Decision: If any ad satisfies both Q1 and Q2, or both Q1 and Q3, then you should determine that "Unexpected Full-Screen Ads" is present.
'''

content = types.Content(
    parts=[
        types.Part(text='Here is a video provided by the user:'),
        types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type='video/mp4')),
        types.Part(text=prompt_introduce_context),
        # types.Part(text=prompt_exemplars),
        # types.Part(inline_data=types.Blob(data=video_bytes_1, mime_type='video/mp4')),
        # types.Part(text=video_introduce_1),
        # types.Part(inline_data=types.Blob(data=video_bytes_2, mime_type='video/mp4')),
        # types.Part(text=video_introduce_2),
        # types.Part(inline_data=types.Blob(data=video_bytes_3, mime_type='video/mp4')),
        # types.Part(text=video_introduce_3),
        # types.Part(inline_data=types.Blob(data=video_bytes_4, mime_type='video/mp4')),
        # types.Part(text=video_introduce_4),
        # types.Part(text=prompt_task),
        types.Part(text=prompt_task_without_exemplars),
    ]
)

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    # model="gemini-2.0-flash",
    contents=content,
    config=config,
)

print(response.text)

