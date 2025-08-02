import cv2
import numpy as np

# --- 全局变量 ---
# 将此路径替换为你的视频文件
VIDEO_PATH = 'E:\\DarkDetection\\dataset\\syx\\us\\En-Part1-396885309.mp4'
frame = None


def mouse_callback(event, x, y, flags, param):
    """鼠标点击回调函数"""
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:  # 检查是否是鼠标左键按下事件
        if frame is not None:
            # 获取点击位置的BGR颜色值
            bgr_color = frame[y, x]

            # 将单个像素的BGR值转换为HSV
            # 注意: cvtColor需要一个数组，所以我们创建一个1x1的图像
            hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

            print("--------------------")
            print(f"Clicked at (x, y): ({x}, {y})")
            print(f"BGR Value: {bgr_color}")
            print(f"HSV Value: H={hsv_color[0]}, S={hsv_color[1]}, V={hsv_color[2]}")

            # 为了方便复制，提供一个建议的范围
            # 对于稳定的颜色，H可以很窄，S和V可以放宽一些以应对光照变化
            h_val = hsv_color[0]

            # 处理红色的Hue环绕问题
            if h_val < 10 or h_val > 170:  # 如果是红色区域
                print("\n--- Suggested HSV Range for RED ---")
                print("# Range 1 (for H around 0)")
                print("lower_red1 = np.array([0, 150, 100])")
                print("upper_red1 = np.array([10, 255, 255])")
                print("\n# Range 2 (for H around 180)")
                print("lower_red2 = np.array([170, 150, 100])")
                print("upper_red2 = np.array([180, 255, 255])")
            else:
                print("\n--- Suggested Generic HSV Range ---")
                print(f"lower = np.array([{max(0, h_val - 10)}, 100, 80])")
                print(f"upper = np.array([{min(180, h_val + 10)}, 255, 255])")


def main():
    """主函数，用于显示视频帧并等待点击"""
    global frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {VIDEO_PATH}")
        return

    # 只读取第一帧进行分析
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧。")
        cap.release()
        return

    window_name = 'Color Picker - Click on the red ring, then press Q to quit'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("工具已启动。请在弹出的窗口中用鼠标左键点击红圈区域。")
    print("控制台将输出该点的HSV值。按 'q' 键退出。")

    while True:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()