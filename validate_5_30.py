from google.genai import types
from google import genai
from typing import Union
import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import argparse

import App_Resumption_Ads
from utils import upload_file, send_request, get_client, dump_upload_files


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

prompt_template = '''
Context:
1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
    a. The persistent red-bordered circle represents the current position of the cursor. 
    b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
    c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

2. The following are some deceptive or malicious UI designs related to advertisements found in apps or ads, which are referred to as "ads dark patterns":
A. "App Resumption Ads": When using an app, users may temporarily exit the app by accessing the iPhone’s Control Center or swiping up to return to the Home Screen. Upon returning to the app, they may be forced to view an advertisement before resuming their activity, disrupting their original experience.
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
L. "Button-Covering Ads": Ads may obscure system or functional buttons (e.g., Home Indicator) within the app, preventing users from interacting with them.
M. "Multiple Close Buttons": Some ads display multiple close buttons simultaneously, making it difficult for users to choose the correct one.
N. "Bias-Driven UI Ads": In ads where users are presented with options, the visual design often emphasizes choices that benefit the advertiser, such as directing users to a landing page or prompting them to watch another ad. 
O. "Disguised Ads": Ads may be designed to closely resemble regular content within apps or the system UI, making them difficult for users to recognize as ads.


Task: Follow the questions below to determine, step by step, whether each type of dark pattern is present:
Step I: Identify all ads in the video:
    QI.1 At what time did advertisements appear (use the format xx:xx–xx:xx)? List all time periods during which ads appeared. Note: (1) Although App Store pages or prompt pages requesting users to rate the app can be part of an ad, they are not considered ad interfaces when standalone. (2) The interface for paid ad removal is part of the app’s functional UI and does not count as an ad. (3) Interfaces that merely request the user to watch an ad (e.g., “Watch an ad for...”) are not considered ads.
    QI.2 Reconsider the ad time intervals you identified---especially those with similar content or consecutive time intervals---and determine whether they represent different stages of the same ad (e.g., a video followed by an playable demo) or different UI components of the same stage (e.g., a full-screen video and a banner with summary information below it). Then treat these intervals as a single ad for further analysis.
Step II: Check the following dark patterns in the entire video:
    1. (Corresponding to "A. App Resumption Ads"):
        Q1: During the app usage, did the user temporarily leave the app? If so, did they return to the Home screen or access the iPhone Control Center? List each pair of timestamps that the user left and returned to the app using the format "xx:xx-xx:xx".
        Q2: If you think Q1 is true, then after the user returned to the app, did an ad you identified in QI.2 immediately appear and occupy most or all of the screen? Note that (1) ads with relatively small size should be disregarded; (2) only ads that pop up within 2 seconds after the user returns to the app can be considered as potential "App Resumption Ads".
        Decision: If any ad satisfies both Q1 and Q2, then you should determine that "App Resumption Ads" is present. List each pair of timestamps that the user left and returned to the app when "App Resumption Ads" occurred.
        {recheck_app_resumption_ads}
    2. (Corresponding to "Unexpected Full-Screen Ads"):
        Q1: Among the ads identified in QI.2, which ones were displayed in full-screen? Note: Ads that initially appear in a non-full-screen format and only expand to full-screen or open a landing page after user interaction should not be counted.
        Q2: Among the ads you identified in Q1, which ones were triggered by the user clicking a normal functional button within the app? Note: "Normal functional buttons" do not include buttons that have indicated “watch ad” through text or icons.
        Q3: Among the ads you identified in Q1, which ones appeared without any gesture input from the user?
        Decision: If any ad satisfies both Q1 and Q2, or both Q1 and Q3, then you should determine that "B. Unexpected Full-Screen Ads" is present.
    3. At what time did the app offer the ad-free service? What does the user need to give or do in order to obtain this service?
        Q1: (Corresponding to "Barter for Ad-Free Privilege") If the user is required to watch ads, rate the app, or perform any action other than making a payment, then "E. Barter for Ad-Free Privilege" presents.
        Q2: (Corresponding to "Paid Ad Removal") If the user is required to make a payment, then "F. Paid Ad Removal" presents.
    4. (Corresponding to "Reward-Based Ads"):
        Q1: Does the app contain buttons that inform the user an ad will appear after clicking? Such buttons may include text like "Watch an ad..." or icons resembling a camera, TV, or video player.
        Q2: Among the buttons you identified in Q1, which buttons indicate the rewards users will receive for watching an ad? Rewards may include in-game currency, items, doubling game earnings, or unlocking new features in the app or game.
        Decision: If you believe the app contains buttons that satisfy both Q1 and Q2, then "G. Reward-Based Ads" presents.
    5. (Corresponding to "Increased Ads with Use"):
        Q1: During app usage, was the app closed and then reopened? Note: "closed" here refers to the app being fully terminated, such as being manually closed by the user in the task manager or crashing, rather than running in the background while the user temporarily leaves.
        Q2: Using the moment identified in Q1 as a dividing point, did the app refrain from showing ads during the first launch but display ads during the second launch?
        Q3: Using the moment identified in Q1 as a dividing point, during the app’s second run, did some UIs display a large number of ads? Did these ad-heavy UIs also contain as many ads during the first run?
        Q4: In terms of overall ad volume, did the app display significantly more ads during the second run compared to the first run?
        Decision: Based on your answers to Q2, Q3, and Q4, determine whether "J. Increased Ads with Use" has occurred. Note: If Q1 indicates that the app was not closed and reopened, then this dark pattern did not present.
Step III: Check the following dark patterns in every ads you listed in QI.2 one by one:
    QIII.1: Which close buttons were shown in the ad, and what were their start and end times (use the format xx:xx-xx:xx)? List all time periods during which ads appeared. Note: In addition to the common “X”, any icon indicating skip, fast-forward, etc., should also be considered a close button.
    QIII.2: Does this ad displayed any landing page? What were their start and end times (use the format xx:xx-xx:xx)? List all time periods during which ads appeared. Note that a landing page can be an app details page in the App Store or a product detail webpage opened in a browser, et al.
    6. (Corresponding to "Auto-Redirect Ads"):
        Q1: For each redirection identified in QIII.2, check whether it was triggered automatically without any user gesture. Conversely, if the user clicked on the ad content or an ad close button just before the redirection occurred, it does not count as an automatic redirection.
        Decision: If the ad satisfies Q1, then you should determine that "C. Auto-Redirect Ads" presents.
    7. (Corresponding to "Ad Without Exit Options"):
        Q1: Check the time intervals identified in QIII.1. Did the earliest close button appear more than three seconds after the ad started?
        Decision: If the ad satisfies Q1, then you should determine that "H. Ad Without Exit Options" presents.
    8. (Corresponding to "Ad Closure Failure"):
        Q1: For the ad close buttons identified in QIII.1, check whether the user clicked them and record the time (use the format xx:xx).
        Q2: For each click in Q1, check whether the ad was not successfully closed. The following behaviors are considered as unsuccessful ad closure: (1) The ad does not respond to the user's click and remains on the current interface. (2) The current interface is closed, but the ad transitions to another interface. (3) The ad redirects to a landing page (e.g., App Store or browser). (4) The current ad is closed but another ad appears immediately (within one second). (5) The ad closes but immediately presents an interface offering an ad removal service.
        Decision: If the ad satisfies Q1 and Q2, then you should determine that "I. Ad Closure Failure" presents.
    9. (Corresponding to "Gesture-Induced Ad Redirection"):
        Q1: In this ad, check whether there are texts or icons prompting the user to shake the phone, such as messages like "Shake your phone for details". Note: Only check whether the text or icon in the ad prompts the user to shake the phone, without evaluating whether the user actually did so or whether it triggered a redirection to a landing page.
        Decision: If the ad satisfies Q1, then you should determine that "K. Gesture-Induced Ad Redirection" presents.
    10. (Corresponding to "Button-Covering Ads"):
        Q1: Is the current ad a non-fullscreen ad? Are any normal functional buttons displayed on the same screen as the ad?
        Q2: If the ad satisfies Q1, does the ad overlap with the system's Home Indicator or with any in-app functional buttons? Note: if the bottom of the ad is aligned with the bottom of the screen, it is considered to overlap with the Home Indicator.
        Q3: If the ad satisfies Q2, did the user attempt to access the Home Indicator or in-app buttons during the ad display but was obstructed by it? Note: You should observe the cursor's movement over a period of time to determine the user's intent---such as the cursor lingering near the button or hovering around it---indicating the user was trying to access that button.
        Decision: If the ad satisfies Q1, Q2, and Q3, then you should determine that "L. Button-Covering Ads" presents.
    11. (Corresponding to "Multiple Close Buttons"):
        Q1: Check the time intervals identified in QIII.1. Do the appearance time intervals of different close buttons in the ad overlap, i.e., do multiple close buttons appear simultaneously?
        Decision: If the ad satisfies Q1, then you should determine that "M. Multiple Close Buttons" presents.
    12. (Corresponding to "Bias-Driven Ads"):
        Q1: Does this ad contain a pair of semantically contrasting buttons, of which one favoring the advertiser (e.g., “View Details”) and the other favoring the user (e.g., “Close Ad”)?
        Q2: If satisfies Q1, are this pair of buttons positioned adjacent to each other, for example, aligned in a row or column, or with their edges close to one another?
        Q3: If satisfies Q2, does the UI design use strong visual contrast to highlight the option that benefits the advertiser?
        Decision: If the ad satisfies Q1, Q2, and Q3, then you should determine that "N. Bias-Driven Ads" presents.
    13. (Corresponding to "Disguised Ads"):
        Q1: Does this ad use UI elements that mimic the operating system’s UI (such as iOS notification pop-ups) in order to obscure its true nature as an ad?
        Decision: If the ad satisfies Q1, then you should determine that "O. Disguised Ads" presents.

Output: List all the ad dark patterns' indices that appeared, separated by spaces. 
Notice:
1. You only need to list the items with indices "A"–"O" regardless of specific manifestations. For instance, you only output "I" even if some instances also match certain manifestations of those dark patterns (e.g., "I.a Multi-Step Ad Closure").
2. If you believe that none of the dark patterns "A"-"O" appear in the video, please output one single letter "P".
Exemplar: After analyzing a video and confirming that it contains "A. App Resumption Ads", "F. Paid Ad Removal", and "I.b Closure Redirect Ads", you output "A F I".
'''

prompt = prompt_template.format(
    recheck_app_resumption_ads=App_Resumption_Ads.recheck_prompt
)

prompt_for_extract_result = '''
Context:
The user is providing you the "result" from Gemini regarding the identification of 15 types of ads dark patterns in a video, indexed by letters "A"-"O". Before presenting its final decision in the last sentence or paragraph, the model may also include an explanation of its reasoning process in this "result".

Task:
You should extract and format the model’s final decision from the "result", and present it in a list including the dark patterns' indices that Gemini has found. All these indices should be separated by spaces.
Notice:
1. Your output should only include the list of indices, separated by spaces, without any other explanation.
2. You should only examine the last sentence or paragraph of the output. If it contains the model’s summary of the dark patterns it identified, extract the dark pattern indices mentioned in that summary and present them in the list of indices.
3. If the last sentence or paragraph of the result indicates that the model found no dark patterns, or if it is "P", then you should simply output "P" to indicate that no dark patterns were present.
4. If the last sentence or paragraph of the result does not summarize the dark pattern indices, you should output "N/A" to indicate that the model did not directly answer my question. You should not infer the dark patterns from the reasoning process, as intermediate steps may contain errors.
Exemplar:
1. Gemini said in the last sentence: "Final list of index letters: B D E F G N", then you output "B D E F G N".
2. Gemini said in the last sentence: "B D I", then you output "B D I".
3. Gemini said in the last sentence: "The identified ad dark patterns are A, B, D, I, L, and N." then you output "A B D I L N"
4. Gemini said in the last sentence: "None of the listed ad dark patterns appeared in the video." then you output "P"
5. Gemini said in the last sentence: "Final list of index letter: P", then you output "P".
6. Gemini said in the last sentence: "P", then you output "P".
'''

index_map = {
    "App Resumption Ads": "A",
    "Unexpected Full-Screen Ads": "B",
    "Auto-Redirect Ads": "C",
    "Long Ad/Many Ads": "D",
    "Barter for Ad-Free Privilege": "E",
    "Paid Ad Removal": "F",
    "Reward-Based Ads": "G",
    "Ad Without Exit Options": "H",
    "Ad Closure Failure": "I",
    "Increased Ads with Use": "J",
    "Gesture-Induced Ad Redirection": "K",
    "Button-Covering Ads": "L",
    "Multiple Close Buttons": "M",
    "Bias-Driven UI Ads": "N",
    "Disguised Ads": "O",
    "No Dark Pattern": "P",
}
# 允许的字母范围：A 到 P
available = ''.join(index_map.values())
all_labels = list(available)


def detection(local_path, app_gt, candidateCount):
    broken_client = False

    result_dict = {}

    client = get_client()
    video_file = upload_file(client, local_path)
    if not video_file:
        result_dict['broken_file_upload'] = True
        return result_dict

    pred_vote = {x: 0 for x in index_map.values()}
    result_dict["candidate"] = []
    for cand in range(candidateCount):
        if broken_client:
            continue

        candidate = {}
        tools = types.Tool(function_declarations=[App_Resumption_Ads.Recheck_App_Resumption_Ads])
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=24576,  # max thinking
            ),
            tools=[tools]
            # candidateCount=5,
        )
        contents = [video_file, prompt]

        # Send request with function declarations
        response = send_request(
            client=client,
            model="gemini-2.5-flash-preview-05-20",
            # contents=[video_file, prompt],
            contents=contents,
            config=config,
        )

        if not response:
            broken_client = True

        candidate["function_call"] = []
        max_calls = 4
        called = 0

        # Check for a function call
        while response and response.function_calls:
            # contents.append(response.candidates[0].content)

            function_calls = response.function_calls
            for function_call in function_calls:
                # print(f"Function to call: {function_call.name}")
                # print(f"Arguments: {function_call.args}")
                called += 1

                func_result = "The function call result is temporarily unavailable."
                if function_call.name == "Recheck_App_Resumption_Ads":
                    new_args = App_Resumption_Ads.get_earlier_latter(local_path, **function_call.args)
                    call_result = App_Resumption_Ads.Actual_Function_Recheck_App_Resumption_Ads(client=client,
                                                                                                video=video_file,
                                                                                                **new_args)
                    if call_result:
                        func_result = call_result
                    else:
                        broken_client = True

                candidate["function_call"].append(
                    {"name": function_call.name, "args": function_call.args, "response": func_result})

                function_response_part = types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": func_result}
                )
                contents.append(types.Content(role='model', parts=[types.Part(function_call=function_call)]))
                contents.append(types.Content(role='user', parts=[function_response_part]))

            if called >= max_calls:
                function_call_config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=24576,  # max thinking
                    ),
                )
            else:
                function_call_config = config

            response = send_request(
                client=client,
                model="gemini-2.5-flash-preview-05-20",
                # contents=[video_file, prompt],
                contents=contents,
                config=function_call_config,
            )
            if not response:
                broken_client = True

        if not response:
            continue
        elif not response.text:
            candidate["finish_reason"] = response.candidates[0].finish_reason.name
            result_dict["candidate"].append(candidate)
            continue

        # print("Here is the response:")
        # print(response.text)

        result = response.text
        candidate["first_response"] = result
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                candidate["thought_summary"] = part.text

        def format_check(str, valid_letters):
            split_str = str.strip().split()
            # 检查每个部分是否是合法大写字母 A-P
            for part in split_str:
                if part not in valid_letters:
                    # print(f"非法字符: '{part}'，模型可能输出了推理过程")
                    return False
            return True

        valid_letters = set(available)

        success = False
        if format_check(result, valid_letters):
            # 按空格分割
            pred_this_candidate = result.strip().split()
            success = True
        else:
            # print('格式检查出错，模型可能输出了推理过程，正在尝试从输出中提取最终结果...')

            extract_response = send_request(
                client=client,
                model="gemini-2.5-flash-preview-05-20",
                config=types.GenerateContentConfig(
                    system_instruction=prompt_for_extract_result),
                contents=result,
            )
            if not extract_response:
                broken_client = True
                continue

            extract_result = extract_response.text
            # print("Here is the extracted result from original result:")
            # print(extract_result)
            candidate["extract_response"] = extract_result

            if extract_result == "N/A":
                # print("Gemini没能从输出中提取到有效的结果，请检查模型输出的最后一段或最后一句是否总结了发现的dark pattern")
                pass
            elif not format_check(extract_result, valid_letters):
                # print("Gemini再次提取的结果格式检查也出错")
                pass
            else:
                pred_this_candidate = extract_result.strip().split()
                success = True

        if success:
            for key_temp in pred_this_candidate:
                pred_vote[key_temp] += 1

        result_dict["candidate"].append(candidate)

    if broken_client:
        result_dict["broken_client"] = True
        return result_dict

    result_dict["pred_vote"] = pred_vote

    pred = []
    for key_temp, value in pred_vote.items():
        if value > candidateCount / 2:
            pred.append(key_temp)

    # 去掉Long Ad/Many Ads选项，暂时不考虑这项DP
    if "D" in pred:
        pred.remove("D")

    gt = [index_map[dp] for dp in app_gt.keys() if dp != "video" and app_gt[dp]]
    # print(f"Ground truth for video {app_gt['video']}:")
    gt_all = ''
    for gt_label in gt:
    #     print(gt_label + ' ', end='')
        gt_all += gt_label + ' '
    # print('\n')
    result_dict["ground_truth"] = gt_all

    # print("Overall, Gemini determines below dark patterns:")
    final_pred = ''
    for item in pred:
    #     print(item + ' ', end='')
        final_pred += item + ' '
    # print('\n')
    result_dict["final_pred"] = final_pred

    def to_binary_vector(label_list, whole_labels):
        return [1 if label in label_list else 0 for label in whole_labels]

    y_true_this_sample = [to_binary_vector(gt, all_labels)]
    y_pred_this_sample = [to_binary_vector(pred, all_labels)]

    return pred, pred_vote, y_true_this_sample, y_pred_this_sample, result_dict

    # print('And Gemini is hesitating on:')
    # for key_temp, value in pred_vote.items():
    #     if value in [math.ceil(candidateCount / 2), math.floor(candidateCount / 2)]:
    #         print(key_temp + ' ', end='')
    # print('\n')

    #
    # precision = precision_score(y_true_this_sample, y_pred_this_sample, average='samples')
    # recall = recall_score(y_true_this_sample, y_pred_this_sample, average='samples')
    # f1 = f1_score(y_true_this_sample, y_pred_this_sample, average='samples')
    #
    # print(f"Precision: {precision:.3f}")
    # print(f"Recall:    {recall:.3f}")
    # print(f"F1-score:  {f1:.3f}")
    # result_dict[local_path]["metrics"] = {'Precision': precision, 'Recall': recall, 'F1-score': f1}
    #
    # y_true.extend(y_true_this_sample)
    # y_pred.extend(y_pred_this_sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='None')
    args = parser.parse_args()

    if args.c != "None":
        with open(args.c, "r") as f:
            old_result_dict = json.load(f)

    # get videos and ground truth
    ground_truth = {}

    gt_videos = "E:\\DarkDetection\\dataset\\syx\\us"
    gt_file = "E:\\DarkDetection\\dataset\\syx\\ground_truth_syx_us.xlsx"
    # 读取 Excel 文件（默认读取第一张表）
    gt_df = pd.read_excel(gt_file)

    # 遍历每一行（从第一行数据开始，不包括表头）
    # 获取所有唯一的序号（按顺序去重）
    unique_ids = gt_df['appid'].drop_duplicates().tolist()

    def unavailable_str(element):
        # 转换为字符串并去除空白
        str_value = str(element).strip()

        # 判断是否是非空字符串
        if str_value in ['', 'nan']:
            return True
        else:
            return False

    for uid in unique_ids[:40]:
        # 找出这个序号对应的所有行
        group = gt_df[gt_df['appid'] == uid]

        # 外层处理：第一行
        first_row = group.iloc[0]
        if first_row.get('能否测试') == '可测试' and glob.glob(os.path.join(gt_videos, f'{uid}-尚宇轩.*')):
            app_gt = {'video': glob.glob(os.path.join(gt_videos, f'{uid}-尚宇轩.*'))[0]}

            Home_Resumption_Ads = True if not unavailable_str(first_row.get('1.1.1home indicator广告标记')) and not unavailable_str(first_row.get('1.1.2发生时间')) else False
            Control_Center_Resumption_Ads = True if not unavailable_str(first_row.get('1.2.1下拉菜单广告标记')) and not unavailable_str(first_row.get('1.2.2发生时间')) else False
            app_gt["App Resumption Ads"] = Home_Resumption_Ads or Control_Center_Resumption_Ads

            Unexpected_Full_Screen_Ads = True if not unavailable_str(first_row.get('1.3.1意外的广告标记')) and not unavailable_str(first_row.get('1.3.2发生时间')) else False
            app_gt["Unexpected Full-Screen Ads"] = Unexpected_Full_Screen_Ads

            Barter_for_Ad_Free_Privilege = True if not unavailable_str(first_row.get('4.2.1免费去广告标记')) and not unavailable_str(first_row.get('4.2.2发生时间')) else False
            app_gt["Barter for Ad-Free Privilege"] = Barter_for_Ad_Free_Privilege

            Paid_Ad_Removal = True if not unavailable_str(first_row.get('4.1.1付费去除广告标记')) and not unavailable_str(first_row.get('4.1.2发生时间')) else False
            app_gt['Paid Ad Removal'] = Paid_Ad_Removal

            Reward_Based_Ads = True if not unavailable_str(first_row.get('6.1通过观看广告获取利益')) and not unavailable_str(first_row.get('6.2发生时间')) else False
            app_gt['Reward-Based Ads'] = Reward_Based_Ads

            Increased_Ads_with_Use = True if not unavailable_str(first_row.get('5.1杀熟，区别对待老用户标记')) and not unavailable_str(first_row.get('5.2发生时间')) else False
            app_gt['Increased Ads with Use'] = Increased_Ads_with_Use

            Auto_Redirect_Ads = False
            Ad_Without_Exit_Option = False
            Ad_Closure_Failure = False
            Gesture_Induced_Ad_Redirection = False
            Button_Covering_Ads = False
            Multiple_Close_Buttons = False
            Bias_Driven_UI_Ads = False
            Disguised_Ads = False

            # 内层处理：所有该序号的行
            for _, row in group.iterrows():
                Auto_Redirect_Ads = Auto_Redirect_Ads or (True if not unavailable_str(row.get('5.1自动跳转的广告标记')) and not unavailable_str(row.get('5.2发生时间.1')) else False)
                Ad_Without_Exit_Option = Ad_Without_Exit_Option or (True if not unavailable_str(row.get('4.4.1没有关闭按钮的广告标记')) and not unavailable_str(row.get('4.4.2发生时间')) else False)
                Ad_Closure_Failure = Ad_Closure_Failure or (True if not unavailable_str(row.get('4.2.1关不掉的广告标记')) and not unavailable_str(row.get('4.2.2发生时间.1')) else False)

                Shake_to_Open = True if not unavailable_str(row.get('3.1"摇一摇"广告标记')) and not unavailable_str(row.get('3.1.1发生时间')) else False
                Hover_to_Open = True if not unavailable_str(row.get('3.3.1遮挡home indicator的广告标记')) and not unavailable_str(row.get('3.3.1发生时间')) else False
                Gesture_Induced_Ad_Redirection = Gesture_Induced_Ad_Redirection or Shake_to_Open or Hover_to_Open

                Home_Indicator_Covering_Ads = True if not unavailable_str(row.get('3.3.1遮挡home indicator的广告标记')) and not unavailable_str(row.get('3.3.1发生时间')) else False
                Other_Button_Covering_Ads = True if not unavailable_str(row.get('3.4.1遮挡按钮的广告标记')) and not unavailable_str(row.get('3.4.2遮挡了什么按钮')) else False
                Button_Covering_Ads = Button_Covering_Ads or Home_Indicator_Covering_Ads or Other_Button_Covering_Ads

                Multiple_Close_Buttons = Multiple_Close_Buttons or (True if not unavailable_str(row.get('4.1关闭按钮设计糟糕的广告标记')) else False)
                Bias_Driven_UI_Ads = Bias_Driven_UI_Ads or (True if not unavailable_str(row.get('6.1.1设计不对称的UI标记')) and not unavailable_str(row.get('6.1.2发生时间')) else False)
                Disguised_Ads = Disguised_Ads or (True if not unavailable_str(row.get('6.2.1伪装成弹窗的UI标记')) and not unavailable_str(row.get('6.2.2发生时间')) else False)

            app_gt["Auto-Redirect Ads"] = Auto_Redirect_Ads
            app_gt["Ad Without Exit Options"] = Ad_Without_Exit_Option
            app_gt["Ad Closure Failure"] = Ad_Closure_Failure
            app_gt["Gesture-Induced Ad Redirection"] = Gesture_Induced_Ad_Redirection
            app_gt["Button-Covering Ads"] = Button_Covering_Ads
            app_gt["Multiple Close Buttons"] = Multiple_Close_Buttons
            app_gt["Bias-Driven UI Ads"] = Bias_Driven_UI_Ads
            app_gt["Disguised Ads"] = Disguised_Ads

            app_gt["Long Ad/Many Ads"] = False

            no_dark_pattern = True
            for key in index_map.keys():
                if key != "No Dark Pattern" and app_gt[key]:
                    no_dark_pattern = False
            app_gt["No Dark Pattern"] = no_dark_pattern

            ground_truth[uid] = app_gt

    ground_truth_temp = ground_truth.copy()
    if args.c != "None":
        for uid, app_gt in ground_truth_temp.items():
            if app_gt["video"] in old_result_dict.keys():
                detect_status = old_result_dict[app_gt["video"]]
                if "broken_client" not in detect_status.keys() and "broken_file_upload" not in detect_status:
                    ground_truth.pop(uid)

    candidateCount = 5
    result_dict = {}
    y_true, y_pred = [], []
    RESULT_LOCK = threading.Lock()

    if args.c != "None":
        result_dict = old_result_dict
        y_true = old_result_dict["y_true"]
        y_pred = old_result_dict["y_pred"]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(detection, app_gt["video"], app_gt, candidateCount): key for key, app_gt in ground_truth.items()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            pred, pred_vote, y_true_this_sample, y_pred_this_sample, result_dict_this_sample = future.result()
            original_key = futures[future]
            app_gt = ground_truth[original_key]
            local_path = app_gt["video"]

            with RESULT_LOCK:
                if "broken_client" in result_dict_this_sample.keys() or "broken_file_upload" in result_dict_this_sample.keys():
                    print(f"Gemini cannot respond with this api on video {app_gt['video']}")
                    result_dict[local_path] = result_dict_this_sample
                    continue

                print(f"Ground truth for video {app_gt['video']}:")
                print(result_dict_this_sample["ground_truth"])

                print("Overall, Gemini determines below dark patterns:")
                print(result_dict_this_sample["final_pred"])

                print('And Gemini is hesitating on:')
                for key_temp, value in pred_vote.items():
                    if value in [math.ceil(candidateCount / 2), math.floor(candidateCount / 2)]:
                        print(key_temp + ' ', end='')
                print('\n')

                precision = precision_score(y_true_this_sample, y_pred_this_sample, average='samples')
                recall = recall_score(y_true_this_sample, y_pred_this_sample, average='samples')
                f1 = f1_score(y_true_this_sample, y_pred_this_sample, average='samples')

                print(f"Precision: {precision:.3f}")
                print(f"Recall:    {recall:.3f}")
                print(f"F1-score:  {f1:.3f}")
                result_dict_this_sample["metrics"] = {'Precision': precision, 'Recall': recall, 'F1-score': f1}

                y_true.extend(y_true_this_sample)
                y_pred.extend(y_pred_this_sample)
                result_dict[local_path] = result_dict_this_sample

    result_dict["y_true"] = y_true
    result_dict["y_pred"] = y_pred

    print("All average metrics:")
    result_dict["all_average_metrics"] = {}

    precision_samples = precision_score(y_true, y_pred, average='samples')
    recall_samples = recall_score(y_true, y_pred, average='samples')
    f1_samples = f1_score(y_true, y_pred, average='samples')
    print('samples:')
    print(f"Average Precision: {precision_samples:.3f}")
    print(f"Average Recall:    {recall_samples:.3f}")
    print(f"Average F1-score:  {f1_samples:.3f}")
    result_dict["all_average_metrics"]["samples"] = {'Precision': precision_samples, 'Recall': recall_samples, 'F1-score': f1_samples}

    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    print('micro:')
    print(f"Average Precision: {precision_micro:.3f}")
    print(f"Average Recall:    {recall_micro:.3f}")
    print(f"Average F1-score:  {f1_micro:.3f}")
    result_dict["all_average_metrics"]["micro"] = {'Precision': precision_micro, 'Recall': recall_micro, 'F1-score': f1_micro}

    precision_per = precision_score(y_true, y_pred, average=None, zero_division=np.nan)
    recall_per = recall_score(y_true, y_pred, average=None, zero_division=np.nan)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=np.nan)
    result_dict["all_average_metrics"]["per_class"] = {}
    for i in range(len(all_labels)):
        label = all_labels[i]
        key = next(k for k, v in index_map.items() if v == label)
        result_dict["all_average_metrics"]["per_class"][key] = {'Precision': precision_per[i], 'Recall': recall_per[i], 'F1-score': f1_per[i]}

    # ===== 重新计算宏平均（忽略无效类）=====
    precision_macro = np.nanmean(precision_per)
    recall_macro = np.nanmean(recall_per)
    f1_macro = np.nanmean(f1_per)
    print('macro:')
    print(f"Average Precision: {precision_macro:.3f}")
    print(f"Average Recall:    {recall_macro:.3f}")
    print(f"Average F1-score:  {f1_macro:.3f}")
    result_dict["all_average_metrics"]["macro"] = {'Precision': precision_macro, 'Recall': recall_macro, 'F1-score': f1_macro}

    dump_upload_files()

    result_file = "result.json"
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    main()
