from google import genai
from google.genai import types
import time

client = genai.Client(api_key='AIzaSyD22lWTZq6SPGgeQicUYic8G1oukNnntxo')

prompt_introduce_context = '''
###
Context:
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
2. The definition of "App Resumption Ads": When using an app, the user may leave the app either actively (e.g., by using the Home Indicator) or passively (e.g., by being redirected to another app such as a browser after clicking somewhere within the app). Upon returning to the app, they may be forced to view an advertisement before resuming their activity, disrupting their original experience.
###
'''
video_local_path = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\App Resumption Ads\\1574455218.mp4"
video_file = client.files.upload(file=video_local_path)
while not video_file.state or video_file.state.name != "ACTIVE":
    print("Processing video...")
    print("File state:", video_file.state)
    time.sleep(5)
    video_file = client.files.get(name=video_file.name)
prompt_task = '''
###
Task: Here is a video provided by the user. Complete following tasks and check if "App Resumption Ads" has occurred in the video:
    Q1: During the app usage, did the user temporarily leave the app? If so, did they return to the Home screen or access the iPhone Control Center? List each pair of timestamps that the user left and returned to the app using the format "xx:xx-xx:xx".
    Q2: If you think Q1 is true, then after the user returned to the app, did an new ad immediately appear and occupy most or all of the screen? Note that (1) ads with relatively small size should be disregarded; (2) only ads that pop up within 2 seconds after the user returns to the app can be considered as potential "App Resumption Ads".
    Decision: If you response "yes" to both Q1 and Q2, then you should determine that "App Resumption Ads" is present. List each pair of timestamps that the user left and returned to the app when "App Resumption Ads" occurred.
###
'''
prompt_exemplars = '''
###
Exemplars: Below are several video cases along with the user’s manual analyses based on the task above. You should learn from these exemplars, and then apply a similar analytical process to the videos provided by the user. When you apply knowledge from an exemplar, you must cite the corresponding exemplar video timestamp and its analysis.
###
'''
video_shot_1 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\App Resumption Ads\\6450351607.mp4"
video_bytes_1 = open(video_shot_1, 'rb').read()
video_introduce_1 = '''
Analysis of Exemplar:
Response to Q1: Yes, the user temporarily left the app at 0:04 and returned immediately at 0:05. Instead of opening the Home Screen or Control Center, they entered the iPhone's app switcher interface. Time interval: 0:04-0:05.
Response to Q2: No. After the user returned to the app, the app immediately displayed a full-screen advertisement. However, upon reviewing the video clip before the user left the app, I found that the advertisement was already playing at the time of departure. Therefore, it does not qualify as a newly triggered full-screen ad.
Decision: Response to Q1 is "yes"; however, response to Q2 is "no", because the ad was not newly displayed after the user returned to the app. Thus, "App Resumption Ads" doesn't occur.
###
'''
video_shot_2 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\App Resumption Ads\\6468838207.mp4"
video_bytes_2 = open(video_shot_2, 'rb').read()
video_introduce_2 = '''
Analysis of Exemplar:
Q1: Yes, the user temporarily left the app at 0:02 and returned at 0:03. At 0:02, the user returned to the Home Screen by using the Home Indicator. Time interval: 0:02-0:03.
Q2: No. After returning to the app, the app displayed a full-screen advertisement at 0:11, but (1) The ad appeared more than 2 seconds after the user returned to the app, so it cannot be determined that the ad was triggered by the app resumption. (2) The user clicked the top-left corner button at 0:06, which directly caused the ad to appear; therefore, the ad was not proactively displayed by the app.
Decision: Q1 is satisfied; however, Q2 is not, because the ad was triggered by the user but not the app. Thus, "App Resumption Ads" doesn't occur.
###
'''
video_shot_3 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\App Resumption Ads\\978674211.mp4"
video_bytes_3 = open(video_shot_3, 'rb').read()
video_introduce_3 = '''
Analysis of  Exemplar:
Q1: Yes, the user temporarily left the app at 0:01 and returned at 0:03. Time interval: 0:01-0:03.
Q2: Yes, after returning to the app, the user was immediately shown a full-screen advertisement. This ad was neither triggered by a user click nor already being displayed at the time the user left the app.
Decision: Both Q1 and Q2 satisfied. Thus, "App Resumption Ads" occurs.
###
'''

content = types.Content(
    parts=[
        types.Part(text=prompt_introduce_context),
        types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type='video/mp4')),
        types.Part(text=prompt_task),
        types.Part(text=prompt_exemplars),
        types.Part(inline_data=types.Blob(data=video_bytes_1, mime_type='video/mp4')),
        types.Part(text=video_introduce_1),
        types.Part(inline_data=types.Blob(data=video_bytes_2, mime_type='video/mp4')),
        types.Part(text=video_introduce_2),
        types.Part(inline_data=types.Blob(data=video_bytes_3, mime_type='video/mp4')),
        types.Part(text=video_introduce_3),
    ]
)

# content = types.Content(
#     parts=[
#         # types.Part(text=prompt_introduce_context),
#         types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type='video/mp4')),
#         # types.Part(text=prompt_task),
#         # types.Part(text=prompt_exemplars),
#         # types.Part(inline_data=types.Blob(data=video_bytes_1, mime_type='video/mp4')),
#         # types.Part(text=video_introduce_1),
#         # types.Part(inline_data=types.Blob(data=video_bytes_2, mime_type='video/mp4')),
#         # types.Part(text=video_introduce_2),
#         # types.Part(inline_data=types.Blob(data=video_bytes_3, mime_type='video/mp4')),
#         # types.Part(text=video_introduce_3),
#     ]
# )

config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,
        thinking_budget=24576,  # max thinking
    ),
    # system_instruction=prompt_introduce_context + prompt_task + prompt_exemplars + video_introduce_1 + video_introduce_2 + video_introduce_3,
)

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    # model="gemini-2.0-flash",
    contents=content,
    config=config,
)

print(response.text)

