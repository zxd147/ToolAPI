import os
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# video_dir = 'bak.web_video'
# json_file = 'bak.video_info.json'
video_dir = '/mnt/digital_service/video/vdmvideo'
json_file = 'video_info.json'

# 读取 JSON 文件
with open(json_file, 'r', encoding='utf-8') as file:
    data = json.load(file)


# 定义检查函数
def check_video_exists(video_name):
    video_path = os.path.join(video_dir, video_name)
    if os.path.exists(video_path):
        return 'exist', video_name, video_path
    else:
        return 'missing', video_name, video_path


# 使用线程池进行并行处理
counter = Counter()
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(check_video_exists, value['video_name']) for value in data.values()]
    for future in futures:
        status, video_name, video_path = future.result()
        counter[status] += 1
        if status == 'exist':
            print(f"视频 {video_name} 存在")
        else:
            print(f"视频 {video_name} 不存在")

# 打印统计结果
print(f'===检查完成===\n{counter["exist"]} 个视频存在, {counter["missing"]} 个视频不存在')
