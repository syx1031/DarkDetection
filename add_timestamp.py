import cv2
import glob
import os

def process_video(input_file, output_file):

    # 打开视频文件
    video_path = input_file
    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 计算时间戳（秒）
        time_in_sec = frame_idx / fps
        timestamp = f"{time_in_sec:.2f}s"

        # 准备文字
        text = f"Frame: {frame_idx} | Time: {timestamp}"

        # 设置文字位置（右下角）
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.4
        thickness = 4
        color = (0, 255, 255)  # 黄色高亮
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        x = width - text_size[0] - 10
        y = height - 10

        # 添加阴影背景（可选）
        cv2.rectangle(frame, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)

        # 绘制文字
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

        # 写入新视频
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    video_files = glob.glob('E:\\DarkDetection\\dataset\\syx\\click\\us\\*')
    for video_file in video_files:
        output_path = os.path.join('E:\\DarkDetection\\dataset\\syx\\frameid\\us', os.path.basename(video_file))
        if not os.path.exists(output_path):
            process_video(video_file, output_path)
