import os
import re
import json
from collections import Counter

from pypinyin import lazy_pinyin, Style


def change_video_name():
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # 循环读取并检查视频路径是否存在
    for index, (key, values) in enumerate(data.items(), start=1):
        if index <= start2:
            if index > start1:
                question_index = '000'
                question = values['question']
                question_letter = text2pinyin(question)
                old_video_name = values['video_name']
                new_video_name = f'{question_index}_{question_letter}.{format}'
                old_video_path = os.path.join(video_dir, old_video_name)
                new_video_path = os.path.join(video_dir, new_video_name)
                if os.path.exists(old_video_path):
                    os.rename(old_video_path, new_video_path)
                    print(f'已成功重命名为 {new_video_name}')
                else:
                    print(f'视频 {old_video_name} 不存在, 将跳过')
                values['video_name'] = new_video_name
            else:
                question_index = key
            if 'question_id' in values:
                # 修改键名和值
                values['question_index'] = values.pop('question_id')  # 更新并移除旧键
            # 创建一个新的字典，将 question_index 放在第一行
            new_values = {
                'question_index': question_index,  # 将 question_index 放在第一行
                **{k: v for k, v in values.items() if k != 'question_index'}  # 添加其他键
            }
            data[key] = new_values  # 更新原字典
            # 如果索引小于14，跳过其余处理
            continue

        question_index = f"{index - start2:03}"
        # index = str(index - start).zfill(3)
        question = values['question']
        question_letter = text2pinyin(question)
        new_video_name = f'{question_index}_{question_letter}.{format}'
        os.rename(os.path.join(video_dir, values['video_name']), os.path.join(video_dir, new_video_name))
        print(f'已成功重命名为 {new_video_name}')
        values['video_name'] = new_video_name
        if 'question_id' in values:
            # 修改键名和值
            values['question_index'] = values.pop('question_id')  # 更新并移除旧键
        # 创建一个新的字典，将 question_index 放在第一行
        new_values = {
            'question_index': question_index,  # 将 question_index 放在第一行
            **{k: v for k, v in values.items() if k != 'question_index'}  # 添加其他键
        }
        data[key] = new_values  # 更新原字典

    # 将修改后的数据保存回文件
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print(f"json数据已成功写回 {json_file}")


def text2pinyin(text):
    # 获取汉字的首字母，包括标点符号
    initials = lazy_pinyin(text, style=Style.FIRST_LETTER)
    # 过滤掉非字母的字符
    initials_filtered = [char for char in initials if re.match(r'[a-zA-Z]', char)]
    initials_str_filtered = ''.join(initials_filtered).upper()
    return initials_str_filtered


if __name__ == "__main__":
    # video_dir = '/mnt/digital_service/video/vdmvideo'
    # json_file = 'video_info.json'
    video_dir = 'bak.web_video'
    json_file = 'bak.video_info.json'
    format = 'mp4'
    start1 = 8
    start2 = 13
    # 使用线程池进行并行处理
    counter = Counter()
    change_video_name()
