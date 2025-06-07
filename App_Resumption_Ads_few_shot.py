from google import genai
from google.genai import types
import time

client = genai.Client(api_key='AIzaSyD22lWTZq6SPGgeQicUYic8G1oukNnntxo')

video_local_path = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\App Resumption Ads\\6459478478.mp4"
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
2. The definition of "App Resumption Ads": When using an app, the user may leave the app either actively (e.g., by using the Home Indicator) or passively (e.g., by being redirected to another app such as a browser after clicking somewhere within the app). Upon returning to the app, they may be forced to view an advertisement before resuming their activity, disrupting their original experience.
'''
prompt_exemplars = '''
Exemplars: There are several exemplars related to "App Resumption Ads", each followed by a detailed analysis:
'''
video_shot_1 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\App Resumption Ads\\6450351607.mp4"
video_bytes_1 = open(video_shot_1, 'rb').read()
video_introduce_1 = '''
Exemplar 1. This is an example where App Resumption Ads did not occur. The video starts and ends with the same full-screen video ad. Although the user temporarily left the app via the iPhone app switcher at 0:04 and returned shortly after, the ad shown after resumption is a continuation of the previous ad, not a new one triggered by returning to the app.
'''
video_shot_2 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\App Resumption Ads\\6468838207.mp4"
video_bytes_2 = open(video_shot_2, 'rb').read()
video_introduce_2 = '''
Exemplar 2. This is an example where App Resumption Ads did not occur. The user left the app at 0:02 and returned at 0:03. At 0:06, the user tapped the button in the top-left corner, which triggered a pop-up saying “Watch Video Ad.” Although the pop-up encourages the user to watch an ad, the pop-up itself is not an ad. Moreover, even if it were considered an ad, it was triggered by the user's intentional interaction, not automatically displayed upon returning to the app.
'''
video_shot_3 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\App Resumption Ads\\978674211.mp4"
video_bytes_3 = open(video_shot_3, 'rb').read()
video_introduce_3 = '''
Exemplar 3: This is an example where App Resumption Ads occurred. The user left the app at 0:01 and returned at 0:03, immediately encountering a full-screen ad. Although a banner ad was shown at the bottom of the screen before the user left, its content differs from the full-screen ad upon return. Moreover, the user did not interact with the screen after returning. These observations indicate that the app automatically presented a new full-screen ad upon resumption.
'''
prompt_task = '''
3. Task: Complete following tasks and check if "App Resumption Ads" has occurred in the video provided by the user:
    Q1: During the app usage, did the user temporarily leave the app? If so, did they return to the Home screen or access the iPhone Control Center? List each pair of timestamps that the user left and returned to the app using the format "xx:xx-xx:xx".
    Q2: If you think Q1 is true, then after the user returned to the app, did an ad you identified in QI.2 immediately appear and occupy most or all of the screen? Note that (1) ads with relatively small size should be disregarded; (2) only ads that pop up within 2 seconds after the user returns to the app can be considered as potential "App Resumption Ads".
    Q3: Discuss the different between Exemplar 1, 2, 3 and user's video.
    Decision: If any ad satisfies both Q1 and Q2, then you should determine that "App Resumption Ads" is present. List each pair of timestamps that the user left and returned to the app when "App Resumption Ads" occurred.
'''
prompt_task_without_exemplars = '''
3. Task: Complete following tasks and check if "App Resumption Ads" has occurred in the video provided by the user:
    Q1: During the app usage, did the user temporarily leave the app? If so, did they return to the Home screen or access the iPhone Control Center? List each pair of timestamps that the user left and returned to the app using the format "xx:xx-xx:xx".
    Q2: If you think Q1 is true, then after the user returned to the app, did an ad you identified in QI.2 immediately appear and occupy most or all of the screen? Note that (1) ads with relatively small size should be disregarded; (2) only ads that pop up within 2 seconds after the user returns to the app can be considered as potential "App Resumption Ads".
    Decision: If any ad satisfies both Q1 and Q2, then you should determine that "App Resumption Ads" is present. List each pair of timestamps that the user left and returned to the app when "App Resumption Ads" occurred.
'''

content = types.Content(
    parts=[
        types.Part(text='Here is a video provided by the user:'),
        types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type='video/mp4')),
        types.Part(text=prompt_introduce_context),
        types.Part(text=prompt_exemplars),
        types.Part(inline_data=types.Blob(data=video_bytes_1, mime_type='video/mp4')),
        types.Part(text=video_introduce_1),
        types.Part(inline_data=types.Blob(data=video_bytes_2, mime_type='video/mp4')),
        types.Part(text=video_introduce_2),
        types.Part(inline_data=types.Blob(data=video_bytes_3, mime_type='video/mp4')),
        types.Part(text=video_introduce_3),
        types.Part(text=prompt_task),
        # types.Part(text=prompt_task_without_exemplars),
    ]
)

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    # model="gemini-2.0-flash",
    contents=content,
    config=config,
)

print(response.text)

