import os
import glob

import cv2
import numpy as np


def find_red_ring(frame, roi_rect=None):
    """
    在给定的帧或ROI中查找红色圆环。
    封装了颜色分割、形态学处理和霍夫圆变换。

    :param frame: 输入的视频帧 (BGR)。
    :param roi_rect: 可选的感兴趣区域 (x, y, w, h)。
    :return: 如果找到圆环，则返回 (x, y, r)；否则返回 None。
    """
    if roi_rect:
        x_roi, y_roi, w_roi, h_roi = roi_rect
        frame_roi = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
        processing_frame = frame_roi
    else:
        processing_frame = frame

    # ... 在 find_red_ring 函数内 ...
    hsv = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2HSV)

    # --- 新的、精确的红色HSV范围 ---
    # 基于测量数据：H=0, S=255, V=170-180
    # 我们只需要一个范围，因为H值稳定在0附近
    lower_red = np.array([0, 240, 170])
    upper_red = np.array([2, 255, 185])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # 2. 形态学处理
    kernel = np.ones((5, 5), np.uint8)
    # 开运算：去除小的噪点
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    # 闭运算：填充圆环内部的空洞
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 为了调试，可以显示掩码
    # cv2.imshow('Debug Mask', red_mask)

    # 3. 圆环检测 (霍夫圆变换)
    # 参数需要根据实际视频中圆环的大小和清晰度进行调整
    circles = cv2.HoughCircles(
        red_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # 累加器分辨率，通常为1-2
        minDist=processing_frame.shape[0] // 4,  # 圆心之间的最小距离，防止检测到多个同心圆
        param1=100,  # Canny边缘检测的高阈值
        param2=25,  # 累加器阈值，越小检测到的圆越多
        minRadius=15,  # 最小半径 (非常重要!)
        maxRadius=80  # 最大半径 (非常重要!)
    )

    if circles is not None:
        # 取第一个检测到的圆
        circle = np.uint16(np.around(circles[0, 0]))
        cx, cy, r = circle[0], circle[1], circle[2]

        # 如果使用了ROI，将坐标转换回全图坐标
        if roi_rect:
            cx += x_roi
            cy += y_roi
        return (cx, cy, r), red_mask  # 同时返回掩码用于调试

    return None, red_mask


def calculate_radius_from_contour(hough_center, mask):
    """
    通过拟合轮廓的最小外接圆来计算半径。

    :param hough_center: Hough变换找到的参考圆心，用于选择正确的轮廓。
    :param mask: 圆环的二值掩码。
    :return: (稳定的中心), 稳定的半径。如果找不到合适的轮廓则返回None, None。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # 选择最大的轮廓，通常就是我们的目标圆环的外轮廓
    # RETR_EXTERNAL 标志使得我们只获取最外层的轮廓，这简化了选择过程。
    target_contour = max(contours, key=cv2.contourArea)

    # 计算该轮廓的最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(target_contour)

    # 返回的是浮点数，可以根据需要转换
    center = (int(x), int(y))
    radius = int(radius)

    return center, radius


def main(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    # --- 状态变量初始化 ---
    R_normal = 27
    # R_normal_buffer = []
    # INIT_FRAMES = 30
    last_known_circle = None
    RADIUS_SHRINK_THRESHOLD = 1
    RADIUS_MININUM = 21
    RADIUS_MAXINUM = 30
    STD_DEV_THRESHOLD = 12.0
    frame_count = 0

    result_stack = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束或读取失败。")
            break

        # 获取当前帧的时间戳（毫秒）
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # 格式化时间戳为 M:S:ms
        seconds = int(timestamp_ms / 1000)
        minutes = seconds // 60
        seconds %= 60
        milliseconds = int(timestamp_ms % 1000)
        timestamp_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

        frame_count += 1
        # 创建用于输出和显示的副本
        display_frame = frame.copy()

        frame_result = None

        # --- 追踪逻辑 ---
        roi_rect = None
        if last_known_circle:
            cx, cy, r = last_known_circle
            roi_size = int(r * 4)
            x_roi = max(0, cx - roi_size // 2)
            y_roi = max(0, cy - roi_size // 2)
            w_roi = min(frame.shape[1] - x_roi, roi_size)
            h_roi = min(frame.shape[0] - y_roi, roi_size)
            roi_rect = (x_roi, y_roi, w_roi, h_roi)
            cv2.rectangle(display_frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (255, 255, 0), 1)

        # 查找圆环
        found_circle_info, debug_mask = find_red_ring(frame, roi_rect)

        # 如果在ROI中没找到，则进行全图搜索作为备用方案
        if found_circle_info is None and last_known_circle is not None:
            found_circle_info, debug_mask = find_red_ring(frame, None)

        # --- 状态判断与更新 ---
        current_state = "Not Detected"
        center_std_dev = 0
        r = 'N/A'  # 预设半径为N/A

        if found_circle_info:
            # 从Hough变换中获取不稳定的结果作为参考
            hough_cx, hough_cy, hough_r_unstable = found_circle_info

            stable_debug_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            # 检查 debug_mask 的尺寸是否与整个帧相同
            if debug_mask.shape == stable_debug_mask.shape:
                # 如果是全尺寸的（来自全图搜索），直接使用
                stable_debug_mask = debug_mask
            elif roi_rect is not None:
                # 如果是小尺寸的（来自ROI搜索），则将其放置在正确的位置
                x_roi, y_roi, w_roi, h_roi = roi_rect
                # 确保 debug_mask 的形状与 roi 区域的形状完全匹配
                if debug_mask.shape == (h_roi, w_roi):
                    stable_debug_mask[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = debug_mask
            debug_mask = stable_debug_mask

            # 2. 使用更稳定的方法计算半径和中心
            #    注意：debug_mask 就是我们需要的二值掩码
            stable_center, stable_radius = calculate_radius_from_contour((hough_cx, hough_cy), debug_mask)

            if stable_radius and RADIUS_MININUM <= stable_radius <= RADIUS_MAXINUM:
                cx, cy, r_val = stable_center[0], stable_center[1], stable_radius
                r = r_val  # 更新半径值
                last_known_circle = (cx, cy, r)

                print(f"[{timestamp_str}] Radius: {r}, cx: {cx}, cy: {cy}, hough_cx: {hough_cx}, hough_cy: {hough_cy}")
                frame_result = {'click': False, 'Radius': r, 'cx': cx, 'cy': cy}

                # if frame_count < INIT_FRAMES:
                #     current_state = "Initializing..."
                #     R_normal_buffer.append(r)
                #     if frame_count == INIT_FRAMES - 1 and R_normal_buffer:
                #         R_normal = np.mean(R_normal_buffer)
                #         print(f"初始化完成, 基准半径 R_normal = {R_normal:.2f}")

                # elif R_normal is not None:
                if R_normal is not None:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    inner_mask = np.zeros(gray_frame.shape, dtype=np.uint8)
                    inner_radius = max(0, r - 5)
                    cv2.circle(inner_mask, (cx, cy), inner_radius, 255, -1)
                    mean, stddev_val = cv2.meanStdDev(gray_frame, mask=inner_mask)
                    center_std_dev = stddev_val[0][0]

                    is_shrunk = r < R_normal - RADIUS_SHRINK_THRESHOLD
                    is_opaque = center_std_dev < STD_DEV_THRESHOLD

                    # if is_shrunk and is_opaque:
                    if is_shrunk:
                        current_state = "Clicked"
                        cv2.circle(display_frame, (cx, cy), r, (0, 255, 0), 2)
                        frame_result['click'] = True
                    else:
                        current_state = "Normal"
                        cv2.circle(display_frame, (cx, cy), r, (255, 0, 255), 2)
                        # R_normal = 0.98 * R_normal + 0.02 * r

                cv2.circle(display_frame, (cx, cy), 2, (0, 0, 255), 3)
            else:
                last_known_circle = None
        else:
            last_known_circle = None

        # --- 在屏幕上显示信息 ---
        info_text = f"State: {current_state}"
        radius_text = f"Radius: {r}" if isinstance(r, int) else f"Radius: {r}"
        r_normal_text = f"R_normal: {R_normal:.2f}" if R_normal else "R_normal: N/A"
        std_dev_text = f"Center StdDev: {center_std_dev:.2f}"

        cv2.putText(display_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, radius_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, r_normal_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, std_dev_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # cv2.imshow('Red Ring Tracker', display_frame)

        # ==========================================================
        #               修正后的调试掩码显示逻辑
        # ==========================================================
        display_mask_to_show = np.zeros(frame.shape[:2], dtype=np.uint8)

        # 检查 debug_mask 是否存在
        if debug_mask is not None:
            # 检查 debug_mask 的尺寸是否与整个帧相同
            if debug_mask.shape == display_mask_to_show.shape:
                # 如果是全尺寸的（来自全图搜索），直接使用
                display_mask_to_show = debug_mask
            elif roi_rect is not None:
                # 如果是小尺寸的（来自ROI搜索），则将其放置在正确的位置
                x_roi, y_roi, w_roi, h_roi = roi_rect
                # 确保 debug_mask 的形状与 roi 区域的形状完全匹配
                if debug_mask.shape == (h_roi, w_roi):
                    display_mask_to_show[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = debug_mask

        # cv2.imshow('Debug Mask', display_mask_to_show)
        # ==========================================================

        result_stack.append(frame_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    idx = 0
    temp_result_stack = result_stack
    while idx < len(result_stack):
        frame_result = result_stack[idx]
        if frame_result and frame_result['click']:
            idx_temp = idx
            while idx_temp > 0:
                idx_temp -= 1
                frame_result_temp = result_stack[idx_temp]
                if not frame_result_temp or frame_result_temp['click'] or frame_result_temp['Radius'] < R_normal:
                    new_frame_result = frame_result_temp if frame_result_temp else result_stack[idx_temp + 1]
                    new_frame_result['click'] = True
                    temp_result_stack[idx_temp] = new_frame_result
                else:
                    break
            idx_temp = idx
            while idx_temp < len(result_stack) - 1:
                idx_temp += 1
                frame_result_temp = result_stack[idx_temp]
                if not frame_result_temp or frame_result_temp['click'] or frame_result_temp['Radius'] < R_normal:
                    new_frame_result = frame_result_temp if frame_result_temp else result_stack[idx_temp - 1]
                    new_frame_result['click'] = True
                    temp_result_stack[idx_temp] = new_frame_result
                else:
                    break
            idx = idx_temp + 1
        else:
            idx += 1
    result_stack = temp_result_stack

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    # 获取视频属性并初始化 VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 使用 'mp4v' 或 'XVID' 编解码器
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束或读取失败。")
            break

        output_frame = frame.copy()

        if result_stack[idx] and result_stack[idx]['click']:
            r, cx, cy = result_stack[idx]['Radius'], result_stack[idx]['cx'], result_stack[idx]['cy']
            diameter = r * 2
            top_left = (cx - r, cy - r)
            bottom_right = (cx + r, cy + r)
            cv2.rectangle(output_frame, top_left, bottom_right, (0, 255, 255), -1)  # -1 表示填充

        # 4. 无论检测结果如何，都将 output_frame 写入视频文件
        video_writer.write(output_frame)

        idx += 1

    # 5. 释放所有资源
    print("处理完成。正在释放资源...")
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"视频已成功保存到: {output_path}")


if __name__ == '__main__':
    video_files = glob.glob('E:\\DarkDetection\\dataset\\syx\\us\\*')
    for video_file in video_files:
        output_path = os.path.join('E:\\DarkDetection\\dataset\\syx\\click\\us', os.path.basename(video_file))
        if not os.path.exists(output_path):
            main(video_file, output_path)
