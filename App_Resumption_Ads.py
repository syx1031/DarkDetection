from google.genai import types
from google import genai
import time
from moviepy import VideoFileClip

import utils
from utils import send_request


# Define a function that the model can call to control smart lights
Recheck_App_Resumption_Ads = {
    "name": "Recheck_App_Resumption_Ads",
    "description": "Provide a more detailed analysis of the events that occurred within 3 seconds before the user left the app and within 3 seconds after they returned to the app. Use start_time as the time when the user left the app and end_time as the time when the user returned.",
    "parameters": {
      "type": "object",
      "properties": {
        "start_time": {
          "type": "string"
        },
        "end_time": {
          "type": "string"
        }
      },
      "required": [
        "start_time",
        "end_time"
      ]
    },
}

recheck_prompt = '''Recheck: If you decide that "App Resumption Ads" occurred, then for each pair of timestamps identified in "Decision", call the function "Recheck_App_Resumption_Ads". Use start_time as the time when the user left the app and end_time as the time when the user returned.
Revise: For each time period in which you requested a function "Recheck_App_Resumption_Ads" call, list both your prior judgment on the presence of App Resumption Ads and the judgment returned by the function call. If there is any discrepancy between the two, you should revise your decision based the result of the function call.
'''

prompt_resumption = '''
Context:
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns darker, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.
2. Ad: any UI interface that promotes a product, service, or app to the user. Note: (1) Although App Store pages or prompt pages requesting users to rate the app can be part of an ad, they are not considered ad interfaces when standalone. (2) The interface for paid ad removal is part of the app’s functional UI and does not count as an ad. (3) Interfaces that merely request the user to watch an ad (e.g., “Watch an ad for...”) are not considered ads.
3. Ad Dark Pattern "App Resumption Ads": When using an app, users may temporarily exit the app by accessing the iPhone’s Control Center or swiping up to return to the Home Screen. Upon returning to the app, they may be forced to view an advertisement before resuming their activity, disrupting their original experience.
4. In our previous discussion, you told me that the user temporarily left the app at {start_time} in the video. They may have returned to the iPhone Home Screen or accessed the Control Center, and then re-entered the app at {end_time}.

Task: Now, I am providing you with a clip of the video from {earlier} to {latter} at {fps} FPS. Please carefully analyze this clip following these tasks and output your analysing process:
Q1. Between {start_time} and {end_time}, did the user indeed leave the current app by returning to the Home Screen, accessing the Control Center, or using the app switcher to switch between apps?
Q2. Between {earlier} and {start_time}, i.e., just before leaving the app, what happened that caused the user to leave? Determine whether the user left the app voluntarily or was forced to leave. Voluntary departure includes intentional interactions with the Home Indicator to open the app switcher, or visits to Control Center. Forced departure occurs when an in-app ad redirects the user to its landing page, during which it opens the App Store or browser. Note that even if the user does click on the ad before it redirects, this event should still be classified as a forced departure.
Q3. Between time {end_time} and {latter}, i.e., after the user returned to the app, check whether a new full-screen ad appeared. Note that if an ad appears but shares the same content and UI as the one shown before the user left the app, it should not be counted as a new full-screen ad.
Decision: Based on Q1, Q2, and Q3, determine whether the video exhibits "App Resumption Ads." This pattern is considered present only if both of the following conditions are met: (1) The user briefly left the app, either voluntarily or was forced to leave. (2) Immediately upon returning to the app, the user was presented with a new full-screen ad.
'''

def get_earlier_latter(local_path, start_time, end_time):
    start_time = utils.time_to_seconds(start_time)
    end_time = utils.time_to_seconds(end_time)

    # 读取视频时长（单位：秒）
    video = VideoFileClip(local_path)
    video_duration = int(video.duration)
    fps = round(video.fps)

    # 计算偏移后时间（注意边界）
    earlier = max(0, start_time - 3)
    latter = min(video_duration, end_time + 3)

    return {'earlier': utils.seconds_to_mmss(earlier), 'start_time': utils.seconds_to_mmss(start_time), 'end_time': utils.seconds_to_mmss(end_time), 'latter': utils.seconds_to_mmss(latter), 'fps': fps}


# This is the actual function that would be called based on the model's suggestion
def Actual_Function_Recheck_App_Resumption_Ads(client: genai.client, video: types.File, earlier: str, start_time: str, end_time: str, latter: str, fps: int) -> str:
    start_offset = str(utils.time_to_seconds(earlier)) + 's'
    end_offset = str(utils.time_to_seconds(latter)) + 's'
    fps = min(24, fps)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
    )

    content = types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri=video.uri, mime_type='video/mp4'),
                video_metadata=types.VideoMetadata(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    fps=fps,
                )
            ),
            types.Part(text=prompt_resumption.format(earlier=earlier, start_time=start_time, end_time=end_time, latter=latter, fps=fps))
        ]
    )

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        # model="gemini-2.0-flash",
        contents=content,
        config=config,
    )

    if not response:
        return None

    return response.text

if __name__ == '__main__':
    video_file = "E:\\DarkDetection\\dataset\\syx\\us\\6450351607-尚宇轩.MP4"

    # Configure the client and tools
    # client = genai.Client(api_key="AIzaSyD6ClWdvvtGbm600-BvopMy4vzkEkqkedI")

    # tools = types.Tool(function_declarations=[Recheck_App_Resumption_Ads])
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        # tools=[tools]
        # candidateCount=5,
    )

    while True:
        try:
            # Send request with function declarations
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                # model="gemini-2.0-flash",
                contents=[video_file, prompt_resumption.format(earlier="xx:xx", start_time="xx:xx", end_time="xx:xx", latter="xx:xx")],
                config=config,
            )
            break
        except Exception as e:
            print('Request failed, try again after 1 min...')
            time.sleep(60)

    print(response.text)
