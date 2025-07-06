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
import traceback

from utils import time_to_seconds, seconds_to_mmss
from Run_Detect import upload_file_and_run_detect


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

    for uid in unique_ids[:40]:
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

            Auto_Redirect_Ads = False
            Auto_Redirect_Ads_per_datapoint = []
            for _, row in group.iterrows():
                Auto_Redirect_Ads = Auto_Redirect_Ads or (True if not unavailable_str(row.get('5.1自动跳转的广告标记')) and not unavailable_str(row.get('5.2发生时间.1')) else False)
                if not unavailable_str(row.get('5.2发生时间.1')):
                    Auto_Redirect_Ads_per_datapoint.append(extract_datapoints(unavailable_str(row.get('5.2发生时间.1'))))

            app_gt["Auto-Redirect Ads"]['video-level'] = Auto_Redirect_Ads
            app_gt['Auto-Redirect Ads']['instance-level'] = Auto_Redirect_Ads_per_datapoint

            Ad_Closure_Failure = False
            Ad_Closure_Failure_per_datapoint = []
            for _, row in group.iterrows():
                Ad_Closure_Failure = Ad_Closure_Failure or (True if not unavailable_str(row.get('4.2.1关不掉的广告标记')) and not unavailable_str(row.get('4.2.2发生时间.1')) else False)
                if not unavailable_str(row.get('4.2.2发生时间.1')):
                    Ad_Closure_Failure_per_datapoint.append(extract_datapoints(unavailable_str(row.get('4.2.2发生时间.1'))))

            app_gt["Ad Closure Failure"]['video-level'] = Ad_Closure_Failure
            app_gt['Ad Closure Failure']['instance-level'] = Ad_Closure_Failure_per_datapoint

            Gesture_Induced = False
            Gesture_Induced_per_datapoint = []
            for _, row in group.iterrows():
                Gesture_Induced = Gesture_Induced or (
                    True if not unavailable_str(row.get('3.1"摇一摇"广告标记')) and not unavailable_str(
                        row.get('3.1.1发生时间')) else False)
                if not unavailable_str(row.get('3.1.1发生时间')):
                    Gesture_Induced_per_datapoint.append(
                        extract_datapoints(unavailable_str(row.get('3.1.1发生时间'))))

            app_gt["Gesture-Induced Ad Redirection"]['video-level'] = Gesture_Induced
            app_gt['Gesture-Induced Ad Redirection']['instance-level'] = Gesture_Induced_per_datapoint

            Multiple_Close_Buttons = False
            Multiple_Close_Buttons_per_datapoint = []
            for _, row in group.iterrows():
                Multiple_Close_Buttons = Multiple_Close_Buttons or (True if not unavailable_str(row.get('4.1关闭按钮设计糟糕的广告标记')) else False)
                if not unavailable_str(row.get('4.1关闭按钮设计糟糕的广告标记')):
                    Multiple_Close_Buttons_per_datapoint.append(extract_datapoints(row.get('2广告出现时间').split('-', 1)[0])[0])

            app_gt["Multiple Close Buttons"]['video-level'] = Multiple_Close_Buttons
            app_gt["Multiple Close Buttons"]['instance-level'] = Multiple_Close_Buttons_per_datapoint

            Ad_Without_Exit_Option = False
            Ad_Without_Exit_Option_per_datapoint = []
            for _, row in group.iterrows():
                Ad_Without_Exit_Option = Ad_Without_Exit_Option or (
                    True if not unavailable_str(row.get('4.4.1没有关闭按钮的广告标记')) and not unavailable_str(row.get('4.4.2发生时间')) else False)
                if not unavailable_str(row.get('4.4.1没有关闭按钮的广告标记')) and not unavailable_str(row.get('4.4.2发生时间')):
                    Ad_Without_Exit_Option_per_datapoint.append(extract_datapoints(row.get('4.4.2发生时间')))

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


def max_matching(prediction, ground_truth, tolerance=5):
    G = nx.Graph()

    pred_secs = [time_to_seconds(t) for t in prediction]
    gt_secs = [time_to_seconds(t) for t in ground_truth]

    # 用索引表示每一个独立的数据点（保留重复）
    pred_nodes = [f"pred_{i}" for i in range(len(pred_secs))]
    gt_nodes = [f"gt_{i}" for i in range(len(gt_secs))]

    # 明确添加节点到图中，并指定 bipartite 属性
    G.add_nodes_from(gt_nodes, bipartite=0)
    G.add_nodes_from(pred_nodes, bipartite=1)

    # 建图（连接所有在 ±tolerance 范围内的边）
    for i, gt_time in enumerate(gt_secs):
        for j, pred_time in enumerate(pred_secs):
            if abs(gt_time - pred_time) <= tolerance:
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
        all_times_str.append(seconds_to_mmss(sec))
        gt_marks.append(1)
        pred_marks.append(1 if i in matched_gt_indices else 0)

    # 加入 prediction 中未匹配上的点
    for j, sec in enumerate(pred_secs):
        if j not in matched_pred_indices:
            all_times_str.append(seconds_to_mmss(sec))
            gt_marks.append(0)
            pred_marks.append(1)

    match_strs = [
        f"{seconds_to_mmss(gt_secs[gi])}->{seconds_to_mmss(pred_secs[pj])}"
        for gi, pj in match_pairs
    ]

    return all_times_str, gt_marks, pred_marks, match_strs


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


def calculate_metrics_on_one_sample(available_dp, ground_truth, prediction):
    metrics = {}
    for key, value in ground_truth.items():
        if key in available_dp:
            all_times_str, gt_label, pred_label, match_strs = max_matching(prediction[key]['instance-level'], value['instance-level'])

            p = precision_score(gt_label, pred_label, zero_division=0)
            r = recall_score(gt_label, pred_label, zero_division=0)
            f1 = f1_score(gt_label, pred_label, zero_division=0)

            metrics[key] = {
                'video-level': {
                    'ground_truth': value['video-level'],
                    'prediction': prediction[key]['video-level'],
                },
                'instance-level': {
                    'timestamp': all_times_str,
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

    video_gt = [result["video-level"]["ground_truth"] for result in metrics.values()]
    video_pred = [result["video-level"]["prediction"] for result in metrics.values()]

    video_precision = precision_score(video_gt, video_pred, zero_division=0)
    video_recall = recall_score(video_gt, video_pred, zero_division=0)
    video_f1 = f1_score(video_gt, video_pred, zero_division=0)

    metrics['overall_metrics_on_video-level_on_this_video'] = {
        "precision": video_precision,
        "recall": video_recall,
        "f1": video_f1,
    }

    return metrics


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


if __name__ == "__main__":
    available_ui = ["Ad", "Recheck Ad"]
    available_dp = ["App Resumption Ads", "Unexpected Full-Screen Ads", "Paid Ad Removal", "Reward-Based Ads", "Auto-Redirect Ads",
                    "Ad Closure Failure", "Gesture-Induced Ad Redirect", "Multiple Close Buttons", "Ad Without Exit Option"]

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
        futures = {executor.submit(upload_file_and_run_detect, app_gt['video'], available_dp): key for key, app_gt in video_need_detect.items()}
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

                metrics = calculate_metrics_on_one_sample(available_dp, app_gt_clean, result_dict_this_sample["prediction"])
                result_dict_this_sample["metrics"] = metrics

                per_ui_metrics = calculate_metrics_per_ui(available_ui, app_gt_clean, result_dict_this_sample)
                result_dict_this_sample["per_ui_metrics"] = per_ui_metrics

    dump_result_file(args.o, result_dict)
