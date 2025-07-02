import argparse
import pandas as pd
import glob
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import networkx as nx
import re
import json
from moviepy import VideoFileClip
import traceback
from pydantic import BaseModel, Field
from google.genai import types

from utils import time_to_seconds, seconds_to_mmss, dump_upload_files, get_client, upload_file, send_request, generate_part
from rag import load_database, generate_video_summarize, generate_exemplars_parts


class AdSegment(BaseModel):
    start_timestamp: str = Field(..., description="The timestamp at which the ad appear: should be represented in the format 'mm:ss'.")
    end_timestamp: str = Field(..., description="The timestamp at which the ad end: should be represented in the format 'mm:ss'.")
    full_screen: bool = Field(..., description="Whether the ad is a full-screen ad at the beginning of it.")
    description: str = Field(..., description="Briefly describe the ad’s position and content in one to two sentences.")
    thinking: str = Field(..., description="Explain in detail the reasoning behind your judgment that an advertisement appeared during this time period.")

    # 自定义验证器确保 period 格式合法
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        if not re.match(r"^\d{2}:\d{2}$", value):
            raise ValueError("timestamp must be in 'mm:ss' format.")
        return value

    def __init__(self, **data):
        # 手动调用验证器
        data['start_timestamp'] = self.validate_timestamp(data['start_timestamp'])
        data['end_timestamp'] = self.validate_timestamp(data['end_timestamp'])
        super().__init__(**data)


# 用于整体结构的 list 类型
AdSegmentList = list[AdSegment]


def detect_ads(client, video):
    prompt_detect_ads = '''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

    Task: 
    1. At what time did advertisements appear (use the format xx:xx–xx:xx)? List all time periods during which ads appeared. Note: (1) Although App Store pages or prompt pages requesting users to rate the app can be part of an ad, they are not considered ad interfaces when standalone. (2) The interface for paid ad removal is part of the app’s functional UI and does not count as an ad. (3) Interfaces that merely request the user to watch an ad (e.g., “Watch an ad for...”) are not considered ads.
    2. Reconsider the ad time intervals you identified in Task 1---especially those with similar content or consecutive time intervals---and determine whether they represent different stages of the same ad (e.g., a video followed by an playable demo) or different UI components of the same stage (e.g., a full-screen video and a banner with summary information below it). Then treat these intervals as a single ad for further analysis. List all time periods during which ads appeared again after your reconsideration.
    '''

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdSegmentList,
        topP=0.01,
        topK=1,
        temperature=0.0,
    )
    contents = [video, prompt_detect_ads]

    # Send request with function declarations
    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        # contents=[video_file, prompt],
        contents=contents,
        config=config,
    )

    # print("Detect Ad:")
    # print(response.text)
    return json.loads(response.text)


retriever = load_database()


def recheck_ads(client, video, start_time, end_time):
    class AdAttribution(BaseModel):
        start_time: str = Field(...,
                                description=f"The timestamp when the ad starts, should be represented in the format 'mm:ss'. If no ad occurs in provided period, set this attribution to '00:00'.")
        end_time: str = Field(...,
                              description=f"The timestamp when the ad ends, should be represented in the format 'mm:ss'. If no ad occurs in provided period, set this attribution to '00:00'.")
        full_screen: bool = Field(..., description="Whether the ad is a full-screen ad at the beginning of it.")
        thinking: str = Field(..., description="Provide detailed analyze about above attributions of the ad.")

        @classmethod
        def validate_timestamp(cls, value):
            if not re.match(r"^\d{2}:\d{2}$", value):
                raise ValueError("timestamp 必须是 mm:ss 格式")
            return value

        def __init__(self, **data):
            # 手动调用验证器
            data['start_time'] = self.validate_timestamp(data['start_time'])
            data['end_time'] = self.validate_timestamp(data['end_time'])
            super().__init__(**data)

    prompt_recheck_ad = f'''
    Context:
    1. Video: This is a clip of screen recording of a user interacting with an app on an iPhone after connecting a mouse.
        a. The persistent red-bordered circle represents the current position of the cursor. 
        b. When the size of the red circle contracts and its center turns black, it indicates that the user is clicking the screen.
        c. Since this is a screen recording, all visible content—except for cursor represented by red circle—reflects what is displayed on the screen rather than any content out of the iPhone's screen.

    Task: An ad may be displayed in the video during the period {start_time}-{end_time}. Please review this period and verify the following attributions of the ad:
    1. Check the few seconds before {start_time} to determine whether the preceding content also belongs to the same ad. Adjust the ad’s start time accordingly based on your analysis. A common scenario is that the app initially displays a non-full-screen ad interface (which contains ad content rather than merely prompting the user to watch an ad). Subsequently, due to user gestures or automatic transitions, the ad switches to a full-screen interface.
    2. Check the few seconds after {end_time} to determine whether the subsequent content also belongs to the same ad. Adjust the ad’s end time accordingly based on your analysis. 
    3. Check the beginning of the ad. If the ad occupies the entire screen or the majority, it should be classified as a "full-screen ad"; conversely, if the ad does not start in full-screen but later becomes full-screen, it should not be classified as a "full-screen ad".
    '''

    ad_summarize = generate_video_summarize(client, video, start_time, end_time)
    retriever_result = retriever.invoke(ad_summarize)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=24576,  # max thinking
        ),
        response_mime_type="application/json",
        response_schema=AdAttribution,
        topP=0.01,
        topK=1,
        temperature=0.0,
    )

    response = send_request(
        client=client,
        model="gemini-2.5-flash-preview-05-20",
        contents=types.Content(
            parts=[generate_part(video.uri, "v"), generate_part(prompt_recheck_ad, "t")] + generate_exemplars_parts(retriever_result),
        ),
        config=config,
    )

    # print("Recheck Ad:")
    # print(response.text)
    return json.loads(response.text), ad_summarize, retriever_result


def unavailable_str(element):
    # 转换为字符串并去除空白
    str_value = str(element).strip()

    # 判断是否是非空字符串
    if str_value in ['', 'nan']:
        return True
    else:
        return False


def extract_datapoints(text):
    # 匹配可能缺前缀0的mm:ss格式（如6:1、06:1、6:01、06:01等），以中英文分号分隔
    pattern = r'(\d{1,2})[:：](\d{1,2})'
    matches = re.findall(pattern, text)

    formatted_times = []
    for minute, second in matches:
        # 转换为标准mm:ss格式
        formatted_time = f'{int(minute):02d}:{int(second):02d}'
        formatted_times.append(formatted_time)

    return formatted_times


def get_ground_truth():
    # get videos and ground truth
    ground_truth = {}

    gt_videos = "E:\\DarkDetection\\dataset\\syx\\us"
    gt_file = "E:\\DarkDetection\\dataset\\syx\\ground_truth_syx_us.xlsx"
    # 读取 Excel 文件（默认读取第一张表）
    gt_df = pd.read_excel(gt_file)

    # 遍历每一行（从第一行数据开始，不包括表头）
    # 获取所有唯一的序号（按顺序去重）
    unique_ids = gt_df['appid'].drop_duplicates().tolist()

    for uid in unique_ids[40:80]:
        # 找出这个序号对应的所有行
        group = gt_df[gt_df['appid'] == uid]

        # 外层处理：第一行
        first_row = group.iloc[0]
        if first_row.get('能否测试') == '可测试' and glob.glob(os.path.join(gt_videos, f'{uid}*')):
            app_gt = {'video': glob.glob(os.path.join(gt_videos, f'{uid}*'))[0]}

            Home_Resumption_Ads = True if not unavailable_str(
                first_row.get('1.1.1home indicator广告标记')) and not unavailable_str(
                first_row.get('1.1.2发生时间')) else False
            Control_Center_Resumption_Ads = True if not unavailable_str(
                first_row.get('1.2.1下拉菜单广告标记')) and not unavailable_str(
                first_row.get('1.2.2发生时间')) else False
            app_gt["App Resumption Ads"] = {'video-level': Home_Resumption_Ads or Control_Center_Resumption_Ads}
            App_Resumption_Ads_per_datapoint = []
            if not unavailable_str(first_row.get('1.1.1home indicator广告标记')) and not unavailable_str(
                    first_row.get('1.1.2发生时间')):
                App_Resumption_Ads_per_datapoint = extract_datapoints(first_row.get('1.1.2发生时间'))
            if not unavailable_str(first_row.get('1.2.1下拉菜单广告标记')) and not unavailable_str(
                    first_row.get('1.2.2发生时间')):
                App_Resumption_Ads_per_datapoint += extract_datapoints(first_row.get('1.2.2发生时间'))
            app_gt["App Resumption Ads"]['instance-level'] = App_Resumption_Ads_per_datapoint

            Unexpected_Full_Screen_Ads = True if not unavailable_str(
                first_row.get('1.3.1意外的广告标记')) and not unavailable_str(first_row.get('1.3.2发生时间')) else False
            app_gt["Unexpected Full-Screen Ads"] = {'video-level': Unexpected_Full_Screen_Ads}
            Unexpected_Full_Screen_Ads_per_datapoint = []
            if not unavailable_str(first_row.get('1.3.1意外的广告标记')) and not unavailable_str(
                    first_row.get('1.3.2发生时间')):
                Unexpected_Full_Screen_Ads_per_datapoint = extract_datapoints(first_row.get('1.3.2发生时间'))
            app_gt['Unexpected Full-Screen Ads']['instance-level'] = Unexpected_Full_Screen_Ads_per_datapoint

            Paid_Ad_Removal = True if not unavailable_str(
                first_row.get('4.1.1付费去除广告标记')) and not unavailable_str(
                first_row.get('4.1.2发生时间')) else False
            app_gt['Paid Ad Removal'] = {'video-level': Paid_Ad_Removal}
            Paid_Ad_Removal_per_datapoint = []
            if not unavailable_str(first_row.get('4.1.1付费去除广告标记')) and not unavailable_str(
                    first_row.get('4.1.2发生时间')):
                Paid_Ad_Removal_per_datapoint = extract_datapoints(first_row.get('4.1.2发生时间'))
            app_gt['Paid Ad Removal']['instance-level'] = Paid_Ad_Removal_per_datapoint

            Reward_Based_Ads = True if not unavailable_str(
                first_row.get('6.1通过观看广告获取利益')) and not unavailable_str(
                first_row.get('6.2发生时间')) else False
            app_gt['Reward-Based Ads'] = {'video-level': Reward_Based_Ads}
            Reward_Based_Ads_per_datapoint = []
            if not unavailable_str(first_row.get('6.1通过观看广告获取利益') and not unavailable_str('6.2发生时间')):
                Reward_Based_Ads_per_datapoint = extract_datapoints(first_row.get('6.2发生时间'))
            app_gt['Reward-Based Ads']['instance-level'] = Reward_Based_Ads_per_datapoint

            # Ads = False if len(group) == 1 else True
            Ads_per_datapoint = []
            for _, row in group.iloc[1:].iterrows():
                raw_timestamp = row.get('2广告出现时间')
                startstamp, endstamp = raw_timestamp.split('-', 1)
                Ads_per_datapoint.append({
                    "start_timestamp": extract_datapoints(startstamp)[0],
                    "end_timestamp": extract_datapoints(endstamp)[0]
                })
            app_gt["Ad"] = Ads_per_datapoint

            ground_truth[uid] = app_gt

    return ground_truth


def max_matching_period(prediction, ground_truth, tolerance=5):
    G = nx.Graph()

    pred_secs = [{'start': time_to_seconds(t["start_timestamp"]), 'end': time_to_seconds(t["end_timestamp"])} for t in prediction]
    gt_secs = [{'start': time_to_seconds(t["start_timestamp"]), 'end': time_to_seconds(t["end_timestamp"])} for t in ground_truth]

    # 用索引表示每一个独立的数据点（保留重复）
    pred_nodes = [f"pred_{i}" for i in range(len(pred_secs))]
    gt_nodes = [f"gt_{i}" for i in range(len(gt_secs))]

    # 明确添加节点到图中，并指定 bipartite 属性
    G.add_nodes_from(gt_nodes, bipartite=0)
    G.add_nodes_from(pred_nodes, bipartite=1)

    # 建图（连接所有在 ±tolerance 范围内的边）
    for i, gt_time in enumerate(gt_secs):
        for j, pred_time in enumerate(pred_secs):
            if abs(gt_time["start"] - pred_time["start"]) <= tolerance and abs(gt_time["end"] - pred_time["end"]) <= tolerance:
                G.add_edge(f"gt_{i}", f"pred_{j}")

    matching = nx.bipartite.maximum_matching(G, top_nodes=gt_nodes)

    matched_gt_indices = set()
    matched_pred_indices = set()
    match_pairs = []

    for gt_node, pred_node in matching.items():
        if gt_node.startswith("gt_"):
            gi = int(gt_node[3:])
            pj = int(pred_node[5:])
            matched_gt_indices.add(gi)
            matched_pred_indices.add(pj)
            match_pairs.append((gi, pj))

    # 生成最终输出列表
    all_times_str = []
    gt_marks = []
    pred_marks = []

    # 保留所有 ground_truth 点
    for i, sec in enumerate(gt_secs):
        all_times_str.append(seconds_to_mmss(sec["start"]) + '-' + seconds_to_mmss(sec["end"]))
        gt_marks.append(1)
        pred_marks.append(1 if i in matched_gt_indices else 0)

    # 加入 prediction 中未匹配上的点
    for j, sec in enumerate(pred_secs):
        if j not in matched_pred_indices:
            all_times_str.append(seconds_to_mmss(sec["start"]) + '-' + seconds_to_mmss(sec["end"]))
            gt_marks.append(0)
            pred_marks.append(1)

    match_strs = [
        f"{seconds_to_mmss(gt_secs[gi]['start'])}-{seconds_to_mmss(gt_secs[gi]['end'])} -> {seconds_to_mmss(pred_secs[pj]['start'])}-{seconds_to_mmss(pred_secs[pj]['end'])}"
        for gi, pj in match_pairs
    ]

    return all_times_str, gt_marks, pred_marks, match_strs


def calculate_metrics_per_ui(available_ui, ground_truth, prediction):
    metrics = {}
    for ui_element in available_ui:
        if ui_element == "Ad":
            ad_gt = ground_truth[ui_element]
            pred = prediction[ui_element]["Result"]
            all_times_str, gt_label, pred_label, match_strs = max_matching_period(pred, ad_gt)

            p = precision_score(gt_label, pred_label, zero_division=0)
            r = recall_score(gt_label, pred_label, zero_division=0)
            f1 = f1_score(gt_label, pred_label, zero_division=0)

            metrics[ui_element] = {
                'video-level': {
                    'ground_truth': True if ad_gt else False,
                    'prediction': True if pred else False,
                },
                'instance-level': {
                    'period': all_times_str,
                    'matched': match_strs,
                    'ground_truth': gt_label,
                    'prediction': pred_label,
                },
                'metrics_on_instance_level_on_this_video': {
                    'precision': p,
                    'recall': r,
                    'f1-score': f1,
                }
            }
        elif ui_element == "Recheck Ad":
            ad_gt = ground_truth["Ad"]

            pred = []
            if prediction["Ad"]["Result"]:
                further_check = prediction["Ad"]["Further Check"]
                if "Recheck Ad" not in further_check.keys():
                    for ad in further_check.values():
                        recheck_result = ad[ui_element]["Result"]
                        if recheck_result:
                            pred.append({
                                "start_timestamp": recheck_result["start_time"],
                                "end_timestamp": recheck_result["end_time"],
                                "full_screen": recheck_result["full_screen"],
                                "thinking": recheck_result["thinking"],
                            })

                    all_times_str, gt_label, pred_label, match_strs = max_matching_period(pred, ad_gt)

                    p = precision_score(gt_label, pred_label, zero_division=0)
                    r = recall_score(gt_label, pred_label, zero_division=0)
                    f1 = f1_score(gt_label, pred_label, zero_division=0)

                    metrics[ui_element] = {
                        'video-level': {
                            'ground_truth': True if ad_gt else False,
                            'prediction': True if pred else False,
                        },
                        'instance-level': {
                            'period': all_times_str,
                            'matched': match_strs,
                            'ground_truth': gt_label,
                            'prediction': pred_label,
                        },
                        'metrics_on_instance_level_on_this_video': {
                            'precision': p,
                            'recall': r,
                            'f1-score': f1,
                        }
                    }

    return metrics


def dump_result_file(path, result_dict):
    with open(path, "w") as f:
        json.dump(result_dict, f, indent=4)


def run_detect(client, video):
    result_dict = {"prediction": {}}

    ads_time = detect_ads(client, video)
    result_dict['Ad'] = {'Result': ads_time}

    result_dict['Ad']['Further Check'] = {}
    for ad in ads_time:
        result_timestamp = start_time = ad["start_timestamp"]
        end_time = ad["end_timestamp"]
        recheck_ads_time, ad_summarize, retriever_results = recheck_ads(client, video, start_time, end_time)
        recheck_ads_time = recheck_ads(client, video, start_time, end_time)
        result_dict["Ad"]['Further Check'][result_timestamp] = {
            'Recheck Ad': {
                'Parameter': [start_time, end_time],
                'ad_summarize': ad_summarize,
                'retriever_results': [json.loads(retriever_result) for retriever_result in retriever_results],
                'Result': recheck_ads_time},
        }

        if not retriever_results:
            pass

    return result_dict


def upload_file_and_run_detect(video_local_path):
    client = get_client(local_path=video_local_path)

    # try:
    #     video_file = upload_file(client, video_local_path)
    # except Exception as e:
    #     return {'broken_file_upload': True, 'error_information': traceback.format_exc()}
    video_file = upload_file(client, video_local_path)
    dump_upload_files()

    # try:
    #     return run_detect(client, video_file)
    # except Exception as e:
    #     return {'broken_client': True, 'error_information': traceback.format_exc()}
    return run_detect(client, video_file)


if __name__ == "__main__":
    available_ui = ["Ad", "Recheck Ad"]

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='None')
    parser.add_argument('-o', type=str, default='result.json')
    args = parser.parse_args()

    if args.c != "None":
        with open(args.c, "r") as f:
            old_result_dict = json.load(f)

    ground_truth = get_ground_truth()

    result_dict = {}
    RESULT_LOCK = threading.Lock()

    if args.c != "None":
        result_dict = old_result_dict

    video_need_detect = dict(list(ground_truth.items()))
    if args.c != "None":
        for uid, app_gt in ground_truth.items():
            if app_gt["video"] in old_result_dict.keys():
                detect_status = old_result_dict[app_gt["video"]]
                if "broken_client" not in detect_status.keys() and "broken_file_upload" not in detect_status and "crashed_future" not in detect_status:
                    video_need_detect.pop(uid)


    # video_need_detect = dict(list(ground_truth.items()))
    # for uid, app_gt in ground_truth.items():
    #     if app_gt["video"] not in [
    #         "E:\\DarkDetection\\dataset\\syx\\us\\6453159988-尚宇轩.MP4",
    #         "E:\\DarkDetection\\dataset\\syx\\us\\1434957889-尚宇轩.mp4"
    #     ]:
    #         video_need_detect.pop(uid)



    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(upload_file_and_run_detect, app_gt['video']): key for key, app_gt in video_need_detect.items()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            original_key = futures[future]
            app_gt = video_need_detect[original_key]
            local_path = app_gt["video"]

            try:
                result_dict_this_sample = future.result()
            except Exception as e:
                print(f"[Error] Future failed on video {local_path}: {e}")
                traceback.print_exc()
                with RESULT_LOCK:
                    result_dict[local_path] = {"crashed_future": True, "error_info": traceback.format_exc()}
                continue

            with RESULT_LOCK:
                if "broken_client" in result_dict_this_sample.keys() or "broken_file_upload" in result_dict_this_sample.keys():
                    print(f"Gemini cannot respond with this api on video {app_gt['video']}")
                    result_dict[local_path] = result_dict_this_sample
                    dump_result_file(args.o, result_dict)
                    continue

                result_dict[local_path] = result_dict_this_sample
                dump_result_file(args.o, result_dict)

    for video_path, result_dict_this_sample in result_dict.items():
        if video_path != "all-average-metrics":
            if all(bad_value not in result_dict_this_sample for bad_value in ['broken_client', 'broken_file_upload', 'crashed_future']):
                app_gt = next((v for k, v in ground_truth.items() if v.get("video") == video_path), None)

                app_gt_clean = app_gt.copy()
                app_gt_clean.pop('video')

                per_ui_metrics = calculate_metrics_per_ui(available_ui, app_gt_clean, result_dict_this_sample)
                result_dict_this_sample["per_ui_metrics"] = per_ui_metrics

    dump_result_file(args.o, result_dict)
