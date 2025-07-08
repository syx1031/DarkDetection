import json

from utils import time_to_seconds, seconds_to_mmss, dump_upload_files, get_client, upload_file
from Detect_Ad import detect_ads, recheck_ads
from Detect_Outside_Interface import detect_outside_interface
from Decide_App_Resumption_Ads import Decide_App_Resumption_Ads
from Detect_Click import detect_click_time_location
from Detect_Hover import detect_hover_time_location
from Detect_Watch_Ad_Text import detect_watch_ad_text_time_location
from Detect_Watch_Ad_Icon import detect_watch_ad_icon_time_location
from Decide_Unexpected_Full_Screen_Ads import Decide_Unexpected_Full_Screen_Ads
from Detect_Watch_Ad_Element import detect_watch_ad_element_time_location
from Detect_Reward_Element import detect_reward_element_time_location
from Decide_Reward_Based_Ads import Decide_Reward_Based_Ads
from Detect_Landing_Page import detect_landing_page_time
from Decide_Auto_Redirect_Ads import Decide_Auto_Redirect_Ads
from Detect_Close_Button import detect_close_button_time_location
from Decide_Ad_Closure_Failure import Decide_Ad_Closure_Failure
from Detect_Shake_Element import detect_shake_element_time_location
from Decide_Gesture_Induced_Ad_Redirection import Decide_Gesture_Induced
from Decide_Ad_Without_Exit_Option import Decide_Ad_Without_Exit_Option
from Decide_Multiple_Close_Buttons import Decide_Multiple_Close_Buttons


def run_detect(client, video, available_dp):
    result_dict = {"prediction": {}}

    result_dict["prediction"]["App Resumption Ads"] = {"video-level": False, 'instance-level': []}
    result_dict["prediction"]["Unexpected Full-Screen Ads"] = {"video-level": False, 'instance-level': []}
    result_dict["prediction"]["Reward-Based Ads"] = {"video-level": False, 'instance-level': []}
    result_dict["prediction"]["Auto-Redirect Ads"] = {"video-level": False, 'instance-level': []}
    result_dict["prediction"]["Ad Closure Failure"] = {"video-level": False, 'instance-level': []}
    result_dict["prediction"]["Gesture-Induced Ad Redirection"] = {"video-level": False, 'instance-level': []}
    result_dict["prediction"]["Ad Without Exit Option"] = {"video-level": False, 'instance-level': []}
    result_dict["prediction"]["Multiple Close Buttons"] = {"video-level": False, 'instance-level': []}

    ads_time = detect_ads(client, video)
    result_dict['Ad'] = {'Result': ads_time}

    result_dict['Ad']['Further Check'] = {}
    for ad in ads_time:
        result_timestamp = start_time = ad["start_timestamp"]
        end_time = ad["end_timestamp"]
        recheck_ads_time, ad_summarize, retriever_results = recheck_ads(client, video, start_time, end_time)
        result_dict["Ad"]['Further Check'][result_timestamp] = {
            'Recheck Ad': {
                'Parameter': [start_time, end_time],
                'ad_summarize': ad_summarize,
                'retriever_results': [json.loads(retriever_result) for retriever_result in retriever_results],
                'Result': recheck_ads_time},
        }

        if "App Resumption Ads" in available_dp and bool(recheck_ads_time["full_screen"]):
            start_time = recheck_ads_time["start_time"]
            end_time = recheck_ads_time["end_time"]
            outside_interface_time = detect_outside_interface(client, video, end_time=start_time)
            result_dict["Ad"]['Further Check'][result_timestamp].update({
                'Outside Interface': {'Parameter': ['end_time=' + start_time], 'Result': outside_interface_time}
            })
            if bool(outside_interface_time["go_outside"]) and time_to_seconds(
                    outside_interface_time["go_outside_time"]) > 0:
                start_time = outside_interface_time["go_outside_time"]
                # Detect "App Resumption Ads"
                App_Resumption_Ads = Decide_App_Resumption_Ads(client, video, recheck_ads_time, outside_interface_time,
                                                               start_time, end_time)
                result_dict["Ad"]['Further Check'][result_timestamp].update({
                    'App Resumption Ads': {'Parameter': [start_time, end_time], 'Result': App_Resumption_Ads}
                })

                if App_Resumption_Ads["app_resumption_ads"]:
                    result_dict["prediction"]["App Resumption Ads"]["video-level"] = True
                    result_dict["prediction"]["App Resumption Ads"]["instance-level"].append(
                        App_Resumption_Ads["ad_start_time"])

        start_time = seconds_to_mmss(max(0, time_to_seconds(recheck_ads_time["start_time"]) - 3))
        end_time = recheck_ads_time["end_time"]
        if "Unexpected Full-Screen Ads" in available_dp and bool(recheck_ads_time["full_screen"]) and time_to_seconds(
                start_time) > 0:
            click_time_location = detect_click_time_location(client, video, start_time, recheck_ads_time["start_time"])
            hover_time_location = detect_hover_time_location(client, video, start_time, recheck_ads_time["start_time"])
            watch_ad_icon_time_location = detect_watch_ad_icon_time_location(client, video, start_time,
                                                                             recheck_ads_time["start_time"])
            watch_ad_text_time_location = detect_watch_ad_text_time_location(client, video, start_time,
                                                                             recheck_ads_time["start_time"])
            # Detect "Unexpected Full-Screen Ads"
            Unexpected_Full_Screen_Ads = Decide_Unexpected_Full_Screen_Ads(client, video, recheck_ads_time,
                                                                           click_time_location, hover_time_location,
                                                                           watch_ad_icon_time_location,
                                                                           watch_ad_text_time_location, start_time,
                                                                           end_time)
            result_dict["Ad"]['Further Check'][result_timestamp].update({
                'Click before Ad': {'Parameter': [start_time, recheck_ads_time["start_time"]],
                                    'Result': click_time_location},
                'Hover before Ad': {'Parameter': [start_time, recheck_ads_time["start_time"]],
                                    'Result': hover_time_location},
                'Watch Ad Icon': {'Parameter': [start_time, recheck_ads_time["start_time"]],
                                  'Result': watch_ad_icon_time_location},
                'Watch Ad Text': {'Parameter': [start_time, recheck_ads_time["start_time"]],
                                  'Result': watch_ad_text_time_location},
                'Unexpected Full-Screen Ads': {'Parameter': [start_time, end_time],
                                               'Result': Unexpected_Full_Screen_Ads},
            })

            if Unexpected_Full_Screen_Ads["unexpected_full_screen_ads"]:
                result_dict["prediction"]["Unexpected Full-Screen Ads"]["video-level"] = True
                result_dict["prediction"]["Unexpected Full-Screen Ads"]["instance-level"].append(Unexpected_Full_Screen_Ads["ad_start_time"])

        if "Auto-Redirect Ads" in available_dp:
            start_time = recheck_ads_time["start_time"]
            end_time = recheck_ads_time["end_time"]

            landing_page_time = detect_landing_page_time(client, video, start_time, end_time)
            result_dict["Ad"]["Further Check"][result_timestamp].update({
                'redirection': {
                    'landing page': landing_page_time
                }
            })

            if landing_page_time["landing_page"]:
                start_time = seconds_to_mmss(max(time_to_seconds(start_time), time_to_seconds(landing_page_time["timestamp"]) - 2))
                end_time = seconds_to_mmss(max(time_to_seconds(start_time) + 1, time_to_seconds(landing_page_time["timestamp"])))
                click_time = detect_click_time_location(client, video, start_time, end_time)
                result_dict["Ad"]["Further Check"][result_timestamp]["redirection"].update({
                    'click before redirection': click_time
                })

                Auto_Redirect_Ads = Decide_Auto_Redirect_Ads(client, video, recheck_ads_time, landing_page_time, click_time)
                result_dict["Ad"]["Further Check"][result_timestamp]["redirection"].update({
                    'Auto-Redirect Ads': Auto_Redirect_Ads
                })

                if Auto_Redirect_Ads["auto_redirect_ads"]:
                    result_dict["prediction"]["Auto-Redirect Ads"]["video-level"] = True
                    result_dict["prediction"]["Auto-Redirect Ads"]["instance-level"].append(Auto_Redirect_Ads["timestamp"])

        close_button = False
        if "Ad Closure Failure" in available_dp:
            start_time = recheck_ads_time["start_time"]
            end_time = recheck_ads_time["end_time"]
            close_button_time_location = detect_close_button_time_location(client, video, start_time, end_time)
            close_button = True
            result_dict["Ad"]["Further Check"][result_timestamp].update({
                'closure failure': {
                    'close button': close_button_time_location
                }
            })

            click_time_location = detect_click_time_location(client, video, start_time, end_time)
            result_dict["Ad"]["Further Check"][result_timestamp]["closure failure"].update({
                'click': click_time_location
            })

            Ad_Closure_Failures = Decide_Ad_Closure_Failure(client, video, recheck_ads_time, close_button_time_location, click_time_location)
            result_dict["Ad"]["Further Check"][result_timestamp]["closure failure"].update({
                'Ad Closure Failure': Ad_Closure_Failures
            })

            for Ad_Closure_Failure in Ad_Closure_Failures:
                if Ad_Closure_Failure["ad_closure_failure"]:
                    result_dict["prediction"]["Ad Closure Failure"]["video-level"] = True
                    result_dict["prediction"]["Ad Closure Failure"]["instance-level"].append(Ad_Closure_Failure["timestamp"])

        if "Gesture-Induced Ad Redirection" in available_dp:
            start_time = recheck_ads_time["start_time"]
            end_time = recheck_ads_time["end_time"]
            shake_element_time_location = detect_shake_element_time_location(client, video, start_time, end_time)
            result_dict["Ad"]["Further Check"][result_timestamp].update({
                'gesture induced': {
                    'shake element': shake_element_time_location
                }
            })

            Gesture_Induced_Ad_Redirection = Decide_Gesture_Induced(client, video, recheck_ads_time, shake_element_time_location)
            result_dict["Ad"]["Further Check"][result_timestamp]["gesture induced"].update({
                'Gesture-Induced Ad Redirection': Gesture_Induced_Ad_Redirection
            })

            if Gesture_Induced_Ad_Redirection["gesture_induced_ad_redirection"]:
                result_dict["prediction"]["Gesture-Induced Ad Redirection"]["video-level"] = True
                result_dict["prediction"]["Gesture-Induced Ad Redirection"]["instance-level"].append(Gesture_Induced_Ad_Redirection["timestamp"])

        if "Multiple Close Buttons" or "Ad Without Exit Option":
            start_time = recheck_ads_time["start_time"]
            end_time = recheck_ads_time["end_time"]
            if not close_button:
                close_button_time_location = detect_close_button_time_location(client, video, start_time, end_time)
            result_dict["Ad"]["Further Check"][result_timestamp].update({
                'multiple (or no) close button': {
                    'close button': close_button_time_location
                }
            })

            Ad_Without_Exit_Option = Decide_Ad_Without_Exit_Option(client, video, recheck_ads_time, close_button_time_location)
            result_dict["Ad"]["Further Check"][result_timestamp]['multiple (or no) close button'].update({
                'Ad Without Exit Option': Ad_Without_Exit_Option
            })

            if Ad_Without_Exit_Option["ad_without_exit_option"]:
                result_dict["prediction"]["Ad Without Exit Option"]["video-level"] = True
                result_dict["prediction"]["Ad Without Exit Option"]["instance-level"].append(Ad_Without_Exit_Option["timestamp"])

            Multiple_Close_Buttons = Decide_Multiple_Close_Buttons(client, video, recheck_ads_time, close_button_time_location)
            result_dict["Ad"]["Further Check"][result_timestamp]['multiple (or no) close button'].update({
                'Multiple_Close_Buttons': Multiple_Close_Buttons
            })

            if Multiple_Close_Buttons["multiple_close_buttons"]:
                result_dict["prediction"]["Multiple Close Buttons"]["video-level"] = True
                result_dict["prediction"]["Multiple Close Buttons"]["instance-level"].append(Multiple_Close_Buttons["timestamp"])

    if "Reward-Based Ads" in available_dp:
        watch_ad_element_time_location = detect_watch_ad_element_time_location(client, video)
        result_dict['watch_ad_element'] = {'Result': watch_ad_element_time_location}

        reward_element_time_location = detect_reward_element_time_location(client, video)
        result_dict['reward_element'] = {'Result': reward_element_time_location}

        Reward_Based_Ads = Decide_Reward_Based_Ads(client, video, watch_ad_element_time_location, reward_element_time_location)
        result_dict["Reward-Based Ads"] = {'Result': Reward_Based_Ads}

        if Reward_Based_Ads["reward_based_ads"]:
            result_dict["prediction"]["Reward-Based Ads"]["video-level"] = True
            result_dict["prediction"]["Reward-Based Ads"]["instance-level"] = Reward_Based_Ads["timestamp"]

    return result_dict


def upload_file_and_run_detect(video_local_path, available_dp):
    # client = get_client(local_path=video_local_path)
    client = get_client(key='AIzaSyA-bxStNkKeadvr4bsd3cd1wBJCpCTrrzA')
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
    return run_detect(client, video_file, available_dp)
