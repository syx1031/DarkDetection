from google import genai
from google.genai import types
import time

client = genai.Client(api_key='AIzaSyD22lWTZq6SPGgeQicUYic8G1oukNnntxo')

image_local_path = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Landing Page\\1475454256.jpg"
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
2. The definition of "Landing Page": A landing page refers to the web page that a user is directed to (may be after clicking or not) on an advertisement, link, or other online source. It can be a standalone page or the homepage of a website, but it is typically designed with a specific marketing goal in mind—such as encouraging the user to register, make a purchase, or download the app.
'''
prompt_exemplars = '''
Exemplars: There are several exemplars related to "landing page", each followed by a detailed analysis:
'''
image_shot_1 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Landing Page\\6453159988.jpg"
image_bytes_1 = open(image_shot_1, 'rb').read()
image_introduce_1 = '''
Exemplar 1. The ad interface shown in this image is not a landing page. Instead, it appears at the end of a video ad as a summary interface, intended to recap the ad’s content for the user. It does not qualify as a landing page because it does not redirect the user to the App Store or a browser, nor does it present detailed information about the advertised product.
'''
image_shot_2 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Landing Page\\6450923985.jpg"
image_bytes_2 = open(image_shot_2, 'rb').read()
image_introduce_2 = '''
Exemplar 2. The ad interface shown in this image is not a landing page. This is a banner that appears during the playback of a video ad, positioned above the video. It summarizes the ad content and attempts to direct the user to the App Store for more information about the app. However, a landing page is typically a full-screen interface that presents additional content not usually included in the ad itself, such as detailed product information. Therefore, this banner does not qualify as a landing page.
'''
image_shot_3 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Landing Page\\1475454256.jpg"
image_bytes_3 = open(image_shot_3, 'rb').read()
image_introduce_3 = '''
Exemplar 3. The ad interface shown in this image is not a landing page. This is a summary page displayed at the end of the ad. Similar to Exemplar 1, this type of page does not redirect the user outside of the ad interface, nor does it provide more detailed information about the advertised product.
'''
image_shot_4 = "E:\\DarkDetection\\Gemini2.5Pro\\few_shot\\Landing Page\\6465991134.jpg"
image_bytes_4 = open(image_shot_4, 'rb').read()
image_introduce_4 = '''
Exemplar 4: The ad interface shown in this image is a landing page. When the user clicked on the ad, it redirected them to the app store page, aiming to persuade them to download the app.
'''
prompt_task = '''
3. Task: Check if the screenshot provided by the user is a landing page, and analyze similarities and differences between Exemplar 1, 2, 3, 4, and user provided screenshot.
'''
prompt_task_without_exemplars = '''
3. Task: Check if the screenshot provided by the user is a landing page.
'''

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

