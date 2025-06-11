from google import genai
from google.genai import types
import time

client = genai.Client(api_key='AIzaSyD22lWTZq6SPGgeQicUYic8G1oukNnntxo')

image_local_path = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Close Button\\1576645378.jpg"
image_local_bytes = open(image_local_path, 'rb').read()

config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,
        thinking_budget=24576,  # max thinking
    ),
)

prompt_introduce_context = '''
Context:
1. Image: This is a screenshot when a user interacted with an app on an iPhone after connecting a mouse.
    a. The red-bordered circle represents the current position of the cursor.
    b. Since this is a screen recording, all visible content (except for cursor represented by red circle) reflects what is displayed on the screen rather than any content out of the iPhone's screen.
2. The definition of "Close Button": The close button refers to any element within an advertisement that clearly signals to the user that clicking it will terminate the current ad. Common close buttons include: (1) Icons indicating the end of a video, such as video skip/fast-forward symbols. (2) Buttons explicitly for closing the ad, such as an “X” icon. (3) Buttons that suggest returning to the app, such as those labeled “Continue to the app”.
'''
prompt_exemplars = '''
Exemplars: There are several exemplars related to "close button", each followed by a detailed analysis:
'''
image_shot_1 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Close Button\\1425445169.jpg"
image_bytes_1 = open(image_shot_1, 'rb').read()
image_introduce_1 = '''
Exemplar 1. The ad interface shown in this image is a close button.
'''
image_shot_2 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Close Button\\1425445169-2.jpg"
image_bytes_2 = open(image_shot_2, 'rb').read()
image_introduce_2 = '''
Exemplar 2. The ad interface shown in this image is a close button.
'''
image_shot_3 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Close Button\\6447110104.jpg"
image_bytes_3 = open(image_shot_3, 'rb').read()
image_introduce_3 = '''
Exemplar 3. The ad interface shown in this image is a close button.
'''
image_shot_4 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Close Button\\6447110104-2.jpg"
image_bytes_4 = open(image_shot_4, 'rb').read()
image_introduce_4 = '''
Exemplar 4: The ad interface shown in this image is a close button.
'''
image_shot_5 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Close Button\\6453159988.jpg"
image_bytes_5 = open(image_shot_5, 'rb').read()
image_introduce_5 = '''
Exemplar 5: The ad interface shown in this image is a close button.
'''
image_shot_6 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Close Button\\1576645378-close-button.jpg"
image_bytes_6 = open(image_shot_6, 'rb').read()
image_introduce_6 = '''
Exemplar 6: The ad interface shown in this image is a close button.
'''
prompt_task = '''
3. Task: Check if the screenshot provided by the user includes close button(s). Please list all close buttons you found.
'''
# prompt_task_without_exemplars = '''
# 3. Task: Check if the screenshot provided by the user is a landing page.
# '''

content = types.Content(
    parts=[
        types.Part(text='Here is a image provided by the user:'),
        types.Part.from_bytes(data=image_local_bytes, mime_type='image/jpeg'),
        types.Part(text=prompt_introduce_context),
        types.Part(text=prompt_exemplars),
        types.Part.from_bytes(data=image_bytes_1, mime_type='image/jpeg'),
        types.Part(text=image_introduce_1),
        types.Part.from_bytes(data=image_bytes_2, mime_type='image/jpeg'),
        types.Part(text=image_introduce_2),
        types.Part.from_bytes(data=image_bytes_3, mime_type='image/jpeg'),
        types.Part(text=image_introduce_3),
        types.Part.from_bytes(data=image_bytes_4, mime_type='image/jpeg'),
        types.Part(text=image_introduce_4),
        types.Part.from_bytes(data=image_bytes_5, mime_type='image/jpeg'),
        types.Part(text=image_introduce_5),
        types.Part.from_bytes(data=image_bytes_6, mime_type='image/jpeg'),
        types.Part(text=image_introduce_6),
        types.Part(text=prompt_task),
    ]
)

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    # model="gemini-2.0-flash",
    contents=content,
    config=config,
)

print(response.text)

