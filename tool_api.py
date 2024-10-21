import asyncio
import base64
import copy
import json
import logging
import mimetypes
import os
import random
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Union

import aiofiles
import httpx
import magic
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

# 支持的音频格式
supported_formats = {'wav', 'mp3', 'pcm', 'flac', 'ogg', 'aac', 'wma', 'm4a', 'raw', 'bytes'}
# 设置输入格式的映射
subtype_format_map = {
    np.float32: 'f32le',  # 输入格式为 32 位浮点小端格式 PCM
    np.float16: 'f16le',
    np.int16: 's16le',
    np.int32: 's32le',
    np.uint8: 'u8',
}
# ffmpeg 命令中的参数, 根据格式匹配音频编码器和输出文件的容器格式
codec_format_map = {
    'wav': {'codec': 'pcm_s16le', 'format': 'wav'},
    'mp3': {'codec': 'libmp3lame', 'format': 'mp3'},
    'pcm': {'codec': 'pcm_s16le', 'format': 'raw'},
    'flac': {'codec': 'flac', 'format': 'flac'},
    'ogg': {'codec': 'libvorbis', 'format': 'ogg'},
    'aac': {'codec': 'aac', 'format': 'adts'},
    'wma': {'codec': 'wmav2', 'format': 'asf'},  # wma或者asf
    'm4a': {'codec': 'aac', 'format': 'mp4'},
    'opus': {'codec': 'libopus', 'format': 'ogg'},  # Opus 格式
    'aiff': {'codec': 'pcm_s16be', 'format': 'aiff'},  # AIFF 格式
    # You can use the format ipod to export to m4a (see original answer) ['matroska', 'mp4', 'ipod']
    'raw': {'codec': 'pcm_s16le', 'format': 'raw'},
    'bytes': {'codec': 'pcm_s16le', 'format': 'raw'},
}
# see original answer: https://stackoverflow.com/questions/62598172/m4a-mp4-audio-file-encoded-with-pydubffmpeg
# -doesnt-play-on-android
# 定义音频格式与其MIME类型的映射
media_type_map = {
    'wav': 'audio/wav',
    'mp3': 'audio/mpeg',  # or 'audio/mp3'
    'pcm': 'audio/pcm',  # 通常PCM格式的MIME类型
    'flac': 'audio/flac',
    'ogg': 'audio/ogg',
    'aac': 'audio/aac',
    'wma': 'audio/wma',  # WMA格式的MIME类型, 'x-ms-wma'
    'm4a': 'audio/m4a',  # M4A格式的MIME类型
    'opus': 'audio/opus',  # Opus格式的MIME类型
    'aiff': 'audio/aiff',  # AIFF格式的MIME类型
    'raw': 'audio/raw',  # PCM格式通常使用audio/pcm，但具体MIME类型可能取决于PCM数据的字节序和位深 ['pcm', 'raw']
    'bytes': 'audio/pcm',  # 通常PCM格式的MIME类型
}


def configure_logging():
    logger = logging.getLogger('tool')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    return logger


tool_logger = configure_logging()
tool_app = FastAPI()
executor = ThreadPoolExecutor(max_workers=3)  # 根据需要设置工作线程数量
tool_app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'],
                        allow_headers=['*'])


class MatchRequest(BaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100))  # 动态生成时间戳
    uid: Union[int, str] = 'admin'
    text: Optional[str] = None  # 支持直接接收文本
    language: str = 'zh'  # 默认语言 "zh"
    initial_prompt: str = '什么是龋齿，牙周炎，智齿，种植牙，蛀牙？'  # 初始提示，中文句子
    content_type: str = 'audio/wav'  # 文件格式默认值 'audio/wav'
    file_format: str = '.wav'  # 默认值 ".wav"后缀
    file_path: Optional[str] = None  # 表单中的文件路径
    file_base64: Optional[str] = None  # 音频 base64 编码
    match_type: str = 'embedding'  # 默认语义检索
    match_threshold: float = 0.85  # 检索阈值


class RetrievalResult(BaseModel):
    query: Optional[str] = None
    retrieved_question: Optional[str] = None
    retrieved_answer: Optional[str] = None
    retrieval_type: Optional[str] = None
    retrieved_score: Optional[float] = None
    match_threshold: Optional[float] = None
    question_id: Optional[str]
    video_name: Optional[str]
    retrieved_response: list = None


class Messages(BaseModel):
    asr_messages: Optional[str]
    asr_result: Optional[str] = None
    retrieval_messages: Optional[str]
    retrieval_result: Optional[RetrievalResult] = None


class Data(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    video_name: str


class MatchResponse(BaseModel):
    code: int
    sno: Optional[Union[int, str]] = None
    data: Optional[Union[Data, str]] = None
    messages: Union[Messages, str]


class ConvertRequest(BaseModel):
    sno: Union[int, str] = Field(default_factory=lambda: int(time.time() * 100))  # 动态生成时间戳
    uid: Union[int, str] = 'admin'
    stream: Optional[bool] = None
    input_format: str = ".mp3"
    audio_format: str = ".wav"  # 默认值 ".wav"后缀
    audio_sampling_rate: int = 16000
    content_type: str = 'audio/wav'  # 文件格式默认值 'audio/wav'
    audio_path: Optional[str] = '/mnt/digital_service/audio/'  # 表单中的文件路径
    audio_base64: Optional[str] = None  # 音频 base64 编码
    return_base64: Optional[bool] = False  # 返回 base64 编码


class ConvertResponse(BaseModel):
    code: int
    sno: Optional[Union[int, str]] = None
    messages: str
    audio_path: Optional[str] = None  # 音频文件路径
    audio_base64: Optional[str] = None  # 音频 base64 编码


# 异步包装器函数，用于运行同步代码
async def run_sync(func, *args):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, func, *args)
    return result


def init_app():
    file_path = 'video_info.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    url = "http://192.168.0.245:3000/api/v1"
    api_key = 'fastgpt-o69bFDwmfF6pMKGI89fsg8VJXJaqvvwFUBLov6a3WVu1UbBt1h1hr76zAT7Ii2U0'
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    client = OpenAI(
        base_url=url,
        api_key=api_key,
    )
    logs = f"Service started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    tool_logger.info(logs)
    return data, client


async def get_audio(audio_file, request_data, output_dir, name):
    audio = None
    audio_info = None
    content_type = request_data['content_type']
    audio_format = request_data['audio_format']
    input_format = request_data['input_format']
    audio_path = request_data['audio_path']
    audio_base64 = request_data['audio_base64']
    # 检查是否至少提供了一个字段
    if not (audio_file or os.path.splitext(audio_path)[1] or audio_base64):
        raise ValueError("ERROR: No audio_file, audio_path, audio_base64 or text provided")
    if audio_file:
        # audio_data = audio_file.file.read()  # 直接使用 `UploadFile` 的文件对象
        audio = await audio_file.read()  # 直接使用 `UploadFile` 的文件对象
        size = len(audio)
        content_type = magic.from_buffer(audio, mime=True) or audio_file.content_type or content_type
        input_format = mimetypes.guess_extension(content_type, strict=False) or input_format
        # audio_path = f'{output_dir}/{name}.{input_format}'
        # # 将音频字节数据保存到文件
        # with open(audio_path, 'wb') as f:
        #     f.write(audio_data)
        # audio_contents.seek(0, 2)  # 移动到文件末尾
        # size = audio_contents.tell()  # 获取当前位置，即文件大小
        # audio_contents.seek(0)  # 将指针重置到文件开头
        audio_info = f"Audio type: file, Format: {input_format}, Size: {size} bytes."
    elif os.path.splitext(audio_path)[1]:  # 是文件路径
        # 从文件路径读取文件内容
        if os.path.exists(audio_path):
            # with open(audio_path, "rb") as f:
            #     audio_data = f.read()  # 读取文件内容
            audio = audio_path
            # 获取文件后缀，例如 ".mp3"
            content_type = magic.from_file(audio_path, mime=True)
            input_format = mimetypes.guess_extension(content_type, strict=False) or input_format or os.path.splitext(audio_path)[1]
            audio_info = f"File type: file_path, Format: {input_format}, Size: unknown bytes."
        else:
            raise FileNotFoundError(f"File not found: {audio_path}.")
    elif audio_base64:
        if audio_base64.startswith("data:"):
            content_type = None
            # 使用正则表达式提取音频格式
            base64_match = re.match(r'data:(.*?);base64,(.*)', audio_base64)
            if base64_match:
                content_type = match.group(1)
                audio_base64 = match.group(2)
        # 解码 base64 编码的音频
        audio = base64.b64decode(audio_base64)
        # 从MIME类型中提取音频格式
        content_type = content_type or magic.from_buffer(audio, mime=True)
        input_format = mimetypes.guess_extension(content_type, strict=False) or input_format
        # audio_path = f'{output_dir}/{name}.{input_format}'
        # with open(audio_path, 'wb') as f:
        #     f.write(audio)
        size = len(audio)
        audio_info = f"File type: file_base64, Format: {audio_format}, Size: {size} bytes."
    if not audio:
        raise ValueError("No valid audio content found.")
    input_format = input_format.replace('.', '')
    return audio_info, input_format, audio


async def get_asr_result(file_name, content_type, file_contents, asr_data):
    asr_url = "http://192.168.0.246:8001/v1/asr"
    headers = {"Content-Type": "multipart/form-data"}
    asr_data['uid'] = 'match_api'
    async with httpx.AsyncClient() as httpx_client:
        response = await httpx_client.post(asr_url, data=asr_data,
                                           files={"audio_file": (file_name, file_contents, content_type)})
        data = response.json()
    return data


async def get_match_result(text, asr_messages, match_data):
    match_type = match_data['match_type']
    match_threshold = match_data['match_threshold']
    # 不允许为空文本，空字符串 ("") 时判断为 False
    if text:
        completion = openai_client.chat.completions.create(
            model="Qwen2-7B-Instruct",
            messages=[
                {"role": "user", "content": text}
            ]
        )
        content = completion.choices[0].message.content
        # 将 JSON 字符串转换为 Python 对象（列表）
        retrieved_response = json.loads(content)
        # 根据检索方式获取 question_id 和对应的 q
        retrieved_result = next(
            ((item['id'], item['q'], item['a'], score['value'])  # 获取 id, q 和 value
             for item in retrieved_response for score in item['score']
             if score['type'] == match_type and score['index'] == 0),
            ('1', 'No matching query', '', 0))
        question_id, retrieved_question, retrieved_answer, retrieved_score = retrieved_result
        # 阈值过滤
        if question_id != '1' and retrieved_score <= match_threshold and match_type == 'embedding':
            code = 1
            question_id = str(random.randint(1, 1))
            video_name = video_info.get(question_id).get('video_name')
            video_text = video_info.get(question_id).get('video_text')
            retrieval_messages = (f'Match conversation responded successfully but the score obtained is too low, '
                                  f'Retrieved text will be empry and default video {video_name} will be use. ')
            retrieval_result = RetrievalResult(
                query=text, retrieved_question=retrieved_question, retrieved_answer=retrieved_answer,
                retrieval_type=match_type, retrieved_score=retrieved_score,
                match_threshold=match_threshold, question_id=question_id, video_name=video_name,
                retrieved_response=retrieved_response)
            match_messages = Messages(asr_messages=asr_messages, asr_result=text, retrieval_messages=retrieval_messages,
                                      retrieval_result=retrieval_result)
            match_data = Data(query=text, question=retrieved_question, video_name=video_name, answer=video_text)
            tool_logger.warning(f'match_data: {match_data}, match_messages: {match_messages}\n')
            return code, match_messages, match_data
        # 如果没有找到符合条件的 ID，question_id为对应ID, code为0
        elif question_id != '1':
            code = 0
            video_name = video_info.get(question_id).get('video_name')
            retrieval_messages = f'Matched and ranked successfully, video {video_name} will be use! '
            retrieval_result = RetrievalResult(
                query=text, retrieved_question=retrieved_question, retrieved_answer=retrieved_answer,
                retrieval_type=match_type, retrieved_score=retrieved_score,
                match_threshold=match_threshold, question_id=question_id, video_name=video_name,
                retrieved_response=retrieved_response)
            match_messages = Messages(asr_messages=asr_messages, asr_result=text, retrieval_messages=retrieval_messages,
                                      retrieval_result=retrieval_result)
            match_data = Data(query=text, question=retrieved_question, video_name=video_name, answer=retrieved_answer)
            tool_logger.info(f'match_data: {match_data}, match_messages: {match_messages}\n')
            return code, match_messages, match_data
        # 如果没有找到符合条件的 ID，question_id为1, code为1
        else:
            code = 1
            question_id = str(random.randint(1, 1))
            video_name = video_info.get(question_id).get('video_name')
            video_text = video_info.get(question_id).get('video_text')
            retrieval_messages = f'No question ID found that matches the provided query: {text}, will use default video {video_name}.\n'
            retrieval_result = RetrievalResult(
                query=text, retrieved_question=retrieved_question, retrieved_answer=retrieved_answer,
                retrieval_type=match_type, retrieved_score=retrieved_score,
                match_threshold=match_threshold, question_id=question_id, video_name=video_name,
                retrieved_response=retrieved_response)
            match_messages = Messages(asr_messages=asr_messages, asr_result=text, retrieval_messages=retrieval_messages,
                                      retrieval_result=retrieval_result)
            match_data = Data(query=text, question=retrieved_question, video_name=video_name, answer=video_text)
            tool_logger.warning(f'match_data: {match_data}, match_messages: {match_messages}\n')
            return code, match_messages, match_data
    # 如果没文本为空，question_id为-1, code为2
    else:
        code = 2
        question_id = str(random.randint(-1, -1))
        video_name = video_info.get(question_id).get('video_name')
        video_text = video_info.get(question_id).get('video_text')
        retrieval_messages = f'No text, the matching function will not be called, and the default video {video_name} will be used. '
        match_messages = Messages(asr_messages=text, retrieval_messages=retrieval_messages)
        match_data = Data(video_name=video_name, answer=video_text)
        tool_logger.error(f'match_data: {match_data}, match_messages: {match_messages}\n')
        return code, match_messages, match_data


async def get_gpt_result(uid, text):
    gpt_url = "http://192.168.0.246:8090/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    gpt_data = {
        "uid": uid,
        "query": text,
        "knowledge": "zyy",
        "project_type": 1
    }
    async with httpx.AsyncClient() as httpx_client:
        response = await httpx_client.post(gpt_url, headers=headers, json=gpt_data)
        data = response.json()
    code = 0
    message = data
    # print("message", message)
    return code, message


async def get_tts_result(uid, text):
    tts_url = "http://192.168.0.246:8088/uptts"
    headers = {"Content-Type": "application/json"}
    tts_data = {
        "uid": uid,
        "content": text
    }
    async with httpx.AsyncClient() as httpx_client:
        response = await httpx_client.post(tts_url, headers=headers, json=tts_data)
        data = response.json()
    code = 0
    message = data
    # print("message", message)
    return code, message


async def go_infer_video(uid, text):
    infer_url = "http://192.168.0.251:5463/inference"
    headers = {"Content-Type": "application/json"}
    infer_data = {
        "uid": uid,
        "audio_name": text,
        "project_type": 1
    }
    async with httpx.AsyncClient() as httpx_client:
        response = await httpx_client.post(infer_url, headers=headers, json=infer_data)
        data = response.json()
    code = 0
    message = data
    # print("message", message)
    return code, message


async def get_text(file, request_data):
    file_name = None
    file_info = None
    file_contents = None
    text = request_data['text']
    content_type = request_data['content_type']
    file_format = request_data['file_format']
    file_path = request_data['file_path']
    file_base64 = request_data['file_base64']
    if text is None:  # 允许空文本的存在，只要发送了text字段
        # 检查是否至少提供了一个字段
        if not (file or file_path or file_base64):
            return JSONResponse(
                content={"code": -1, "message": "ERROR: No file, file path, base64 audio or text provided"},
                status_code=400
            )
        if file:
            # 直接使用 UploadFile 对象
            file_name = file.filename or str(uuid.uuid4().hex[:8])
            content_type = file.content_type or content_type
            # 如果没有后缀名，尝试根据 MIME 类型添加后缀
            if '.' not in file_name:
                # content_type = 'audio/mpeg'
                guess_format = mimetypes.guess_extension(content_type, strict=False)
                extension = guess_format or file_format
                file_name += extension
            file_contents = await file.read()
            file_info = f"File type: file, File name: {file_name}, Content_type: {content_type}, Size: {len(file_contents)} bytes."
        elif file_path:
            # 从文件路径读取文件内容
            file_name = os.path.basename(file_path)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    file_contents = f.read()
                file_info = f"File type: file_path, File path: {file_path}, Size: {len(file_contents)} bytes."
            else:
                raise FileNotFoundError(f"File not found: {file_path}.")
        elif file_base64:
            # 解码 base64 编码的音频
            file_name = str(uuid.uuid4().hex[:8]) + file_format
            file_contents = base64.b64decode(file_base64)
            file_info = f"File type: file_base64, File name: {file_name}, Size: {len(file_contents)} bytes."
        if file_contents is None:
            raise ValueError("No valid file content found.")
        asr_data = await get_asr_result(file_name, content_type, file_contents, request_data)
        text = asr_data.get('text')
        asr_messages = asr_data.get('messages').rstrip()
    else:
        file_info = f"File type: text, File name: null, Text content: {text}."
        asr_messages = 'Text exists and will be used directly instead of ASR '
    tool_logger.info(f'asr: asr_messages: {asr_messages}, asr_result: {text}')
    return text, file_info, asr_messages


async def write_input(process, audio_buffer):
    chunk_size = 1024
    for i in range(0, len(audio_buffer), chunk_size):
        process.stdin.write(audio_buffer[i:i + chunk_size])
        await process.stdin.drain()  # 等待缓冲区有空间
    process.stdin.close()  # 写入完毕后关闭标准输入


async def stream_output(process):
    while True:
        audio = await process.stdout.read(1024)
        if not audio:
            break
        yield audio


async def convert_by_ffmpeg_on_buffer(audio, request_data, stream, input_format):
    start = time.process_time()
    # 使用ffmpeg, 生成器流式写入音频数据并读取输出
    audio_format = request_data['audio_format'].replace('.', '')
    audio_sampling_rate = request_data['audio_sampling_rate']
    codec = codec_format_map[audio_format]['codec']
    # 设置环境变量（如果需要的话）
    os.environ['FFMPEG_BUFFER_SIZE'] = str(1024 * 1024 * 3)  # 设置为1MB
    # 使用 ffmpeg-python 进行流式处理
    # 写入部分
    command = [
        'ffmpeg',
        '-stream_loop', '-1',
        '-i', "pipe:0",  # 从文件路径读取, 直接开始处理数据
        '-f', input_format,
        '-ar', str(audio_sampling_rate),
        '-c:a', codec,  # 指定音频编解码器
        '-ac', '2',
        '-vn',  # 不处理视频
        '-async', '1',
        '-vsync', '1',
        '-threads', '4',
        '-f', audio_format,
        'pipe:1'  # 输出到标准输出
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    # 启动 write_task 后，事件循环会立即返回，并允许执行后续的代码，
    write_task = asyncio.create_task(write_input(process, audio))
    if stream:
        try:
            # 流式情况, 从子进程每次读取1024字节
            async for audio_data in stream_output(process):
                yield audio_data
            await write_task
            # for audio_data in iter(lambda: process.stdout.read(1024), b''):  # 同步情况
            #     yield audio_data  # 确保返回字节数据
        finally:
            # finally 块中的代码无论是否捕获到异常都会执行。
            # 确保在结束时关闭子进程，避免资源泄漏
            await process.wait()
            tool_logger.info('done')
    else:
        # process.stdin.close()  # 关闭标准输入以表示结束
        # audio_data = await process.stdout.read()
        audio_data, stderr = await process.communicate()
        yield audio_data
        await process.wait()  # 等待进程完成
    end = time.process_time()
    tool_logger.info(f"time_all: {end - start}")
    # 检查错误信息
    if process.returncode != 0:
        error_data = process.stderr.read()
        raise RuntimeError(f'FFmpeg error: {error_data}')


async def convert_by_ffmpeg_on_path(audio, request_data, stream, input_format):
    start = time.process_time()
    # 使用ffmpeg, 生成器流式写入音频数据并读取输出
    audio_format = request_data['audio_format'].replace('.', '')
    audio_sampling_rate = request_data['audio_sampling_rate']
    codec = codec_format_map[audio_format]['codec']
    # 创建 FFmpeg 进程, ffmpeg 直接从文件路径读取音频数据，这个过程流式的，因为 ffmpeg 可以在读取文件的同时处理数据。
    # 设置环境变量（如果需要的话）
    os.environ['FFMPEG_BUFFER_SIZE'] = str(1024 * 1024 * 3)  # 设置为1MB
    # 写入部分
    command = [
        'ffmpeg',
        '-i', audio,  # 从文件路径读取, 直接开始处理数据
        '-f', input_format,
        '-ar', str(audio_sampling_rate),
        '-c:a', codec,  # 指定音频编解码器
        '-ac', '2',
        '-vn',  # 不处理视频
        '-async', '1',
        '-vsync', '1',
        '-threads', '4',
        '-f', audio_format,
        'pipe:1'  # 输出到标准输出
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    if stream:
        try:
            process.stdin.close()  # 关闭标准输入以表示结束
            # 流式情况, 从子进程每次读取1024字节
            async for audio_data in stream_output(process):
                yield audio_data
            # for audio_data in iter(lambda: process.stdout.read(1024), b''):
            #     yield audio_data  # 确保返回字节数据
        finally:
            # finally 块中的代码无论是否捕获到异常都会执行。
            # 确保在结束时关闭子进程，避免资源泄漏
            await process.wait()
            tool_logger.info('done')
    else:
        # 非流式情况，读取所有输出数据
        # process.stdin.close()  # 关闭标准输入以表示结束
        # audio_data = await process.stdout.read()
        audio_data, stderr = await process.communicate()
        yield audio_data  # 返回所有输出数据
        await process.wait()  # 等待进程完成
    end = time.process_time()
    tool_logger.info(f"time_all: {end - start}")
    # 检查错误信息
    if process.returncode != 0:
        error_data = process.stderr.read()
        raise RuntimeError(f'FFmpeg error: {error_data}')


@tool_app.middleware("http")
async def log_requests(request: Request, call_next):
    logs = "Request arrived"
    tool_logger.debug(logs)
    response = await call_next(request)
    return response


@tool_app.get("/")
async def index():
    service_name = """
        <html> <head> <title>tts_service</title> </head>
            <body style="display: flex; justify-content: center;"> <h1> tool_api</h1></body> </html>
        """
    return HTMLResponse(content=service_name, status_code=200)


@tool_app.get("/health")
async def health():
    """Health check."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    health_data = {"status": "healthy", "timestamp": timestamp}
    # 返回JSON格式的响应
    return JSONResponse(content=health_data, status_code=200)


@tool_app.post("/conversation/match")
@tool_app.post("/v1/match")
async def match(
        request: Request,
        file: Optional[UploadFile] = File(None)  # 上传的文件
):
    try:
        # 判断请求的内容类型
        if request.headers.get('content-type') == 'application/json':
            json_data = await request.json()
            request = MatchRequest(**json_data)
        else:
            # 解析表单数据
            form_data = await request.form()
            request = MatchRequest(**form_data)
        request_data = request.model_dump()
        sno = request_data['sno']
        text, file_info, asr_messages = await get_text(file, request_data)
        # 准备要发送的数据
        logs = f"Completions request param: {request_data}, file_info: {file_info}"
        tool_logger.info(logs)
        # code, messages, answers = await run_sync(get_retrieval_result, text)  # 将同步函数放到异步线程
        code, messages, data = await get_match_result(text, asr_messages, request_data)
        results = MatchResponse(
            code=code,
            sno=sno,
            messages=messages,
            data=data
        )
        logs = f"Completions response results: {results}\n"
        tool_logger.info(logs)
        return JSONResponse(status_code=200, content=results.model_dump())
    except json.JSONDecodeError as je:
        error_message = MatchResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Completions response  error: {error_message}\n "
        tool_logger.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    except Exception as e:
        error_message = MatchResponse(
            code=-1,
            messages=f"Exception: {str(e)}"
        )
        logs = f"Completions response error: {error_message}\n"
        tool_logger.error(logs)
        raise HTTPException(status_code=500, detail=error_message.model_dump())


@tool_app.post("/conversation/live")
@tool_app.post("/v1/match1")
async def live(
        request: Request,
        file: Optional[UploadFile] = File(None)  # 上传的文件
):
    try:
        # 判断请求的内容类型
        if request.headers.get('content-type') == 'application/json':
            json_data = await request.json()
            request = MatchRequest(**json_data)
        else:
            # 解析表单数据
            form_data = await request.form()
            request = MatchRequest(**form_data)
        if request.uid != "dentist":
            results = MatchResponse(
                code=-2,
                sno=request.sno,
                messages="用户名不对",
            )
            return JSONResponse(status_code=200, content=results.model_dump())
        request_data = request.model_dump()
        sno = request_data['sno']
        text, file_info, asr_messages = await get_text(file, request_data)
        # 准备要发送的数据
        logs = f"Completions request param: {request_data}, file_info: {file_info}"
        tool_logger.info(logs)
        code, messages = await get_gpt_result(request.uid, text)
        sRet = "ok"
        for idx, msg in enumerate(messages['answers']):
            logs = f"answer_{idx + 1} = {msg}\n"
            tool_logger.info(logs)
            code, audio_data = await get_tts_result(request.uid, msg)
            audio_name = audio_data["return_path"].split("/")[1]
            logs = f"tts results: {audio_name}\n"
            tool_logger.info(logs)
            code, info = await go_infer_video(request.uid, audio_name)
            if info["status"] == 3:
                code = 0
            else:
                code = info["status"]
                sRet = "推理失败"
            logs = f"inference result: {code}, info: {info}\n"
            # answers = messages['answers'][0]
            tool_logger.info(logs)
        results = MatchResponse(
            code=code,
            sno=sno,
            messages=sRet,
        )
        logs = f"Completions response results: {results}\n"
        tool_logger.info(logs)
        return JSONResponse(status_code=200, content=results.model_dump())
    except json.JSONDecodeError as je:
        error_message = MatchResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Completions response  error: {error_message}\n "
        tool_logger.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    except Exception as e:
        error_message = MatchResponse(
            code=-1,
            messages=f"Exception: {str(e)}"
        )
        logs = f"Completions response error: {error_message}\n"
        tool_logger.error(logs)
        raise HTTPException(status_code=500, detail=error_message.model_dump())


@tool_app.post("/audio/convert")
async def convert_audio(
        request: Request,
        audio_file: UploadFile = File(None),
):
    try:
        start = time.process_time()
        # 判断请求的内容类型
        if request.headers.get('content-type') == 'application/json':
            json_data = await request.json()
            request = ConvertRequest(**json_data)
        else:
            # 解析表单数据
            form_data = await request.form()
            request = ConvertRequest(**form_data)
        request_data = request.model_dump()
        sno = request_data['sno']
        uid = request_data['uid']
        stream = request_data['stream']
        audio_path = request_data['audio_path']
        audio_format = request_data['audio_format']
        return_base64 = request_data['return_base64']
        name = str(uuid.uuid4().hex[:8])
        is_audio_file = bool(audio_file)
        is_audio_path = bool(os.path.splitext(audio_path)[1])
        stream = is_audio_file if stream is None else stream
        audio_name = os.path.splitext(os.path.basename(audio_path))[0] if is_audio_path else uid
        output_dir = os.path.dirname(audio_path) if is_audio_path else audio_path
        output_path = os.path.join(output_dir, f'{audio_name}_{name}{audio_format}')
        audio_format = audio_format.replace('.', '')
        media_type = media_type_map[audio_format]
        audio_info, input_format, audio = await get_audio(audio_file, request_data, output_dir, name)
        logs = f"Convert audio request param: stream: {stream}, sno={sno}, uid={uid}, input_format={input_format}, to audio_format={audio_format}, audio info: {audio_info} "
        tool_logger.info(logs)
        convert_by_ffmpeg_map = {0: convert_by_ffmpeg_on_buffer, 1: convert_by_ffmpeg_on_path}
        audio_generator = convert_by_ffmpeg_map[is_audio_path](audio, request_data, stream, input_format)
        if stream:
            return StreamingResponse(audio_generator, media_type=media_type)
        else:
            # audio_data = await audio_generator.__anext__()  # .__anext__() 是异步迭代器对象的一个方法，它用于获取异步生成器的下一个项。
            # 创建一个异步任务来迭代生成器
            # task = asyncio.create_task(audio_generator.__anext__())
            # # 等待任务完成并获取结果
            # audio_data = await task
            # 使用 run_sync 将同步的 audio_generator 扔到线程池中执行
            # audio_data = list(audio_generator)[0]
            # audio_data = await run_sync(lambda: list(audio_generator)[0])
            # audio_data = await run_sync(lambda: next(audio_generator))
            audio_data = await audio_generator.__anext__()  # .__anext__() 是异步迭代器对象的一个方法，它用于获取异步生成器的下一个项。
            # 写入到文件
            async with aiofiles.open(output_path, mode='wb') as file:
                await file.write(audio_data)
            # 将音频内容编码为 base64
            data_url = f"data:{media_type};base64"
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            audio_base64 = f'{data_url},{audio_base64}'
            # len('data:audio/wav;base64,'): 22
            audio_base64_log = audio_base64[:30] + "..." + audio_base64[-20:]  # 只记录前30个字符
            if not return_base64:
                audio_base64 = audio_base64_log
        messages = f"Audio convert successfully!"
        results = ConvertResponse(
            code=0,
            sno=sno,
            messages=messages,
            audio_path=output_path,
            audio_base64=audio_base64
        )
        results_log = copy.deepcopy(results)
        results_log.audio_base64 = audio_base64_log
        logs = f"Audio convert results: {results_log}\n"
        tool_logger.info(logs)
        end = time.process_time()
        tool_logger.info(f"time_all: {end - start}")
        return JSONResponse(status_code=200, content=results.model_dump())
    except json.JSONDecodeError as je:
        error_message = ConvertResponse(
            code=-1,
            messages=f"JSONDecodeError, Invalid JSON format: {str(je)} "
        )
        logs = f"Audio convert error: {error_message}\n "
        tool_logger.error(logs)
        return JSONResponse(status_code=400, content=error_message.model_dump())
    # except Exception as e:
    #     error_message = ConvertResponse(
    #         code=-1,
    #         messages=f"Exception: {str(e)}"
    #     )
    #     logs = f"Audio convert error: {error_message}\n"
    #     tool_logger.error(logs)
    #     raise HTTPException(status_code=500, detail=error_message.model_dump())


@tool_app.get('/audio/convert', response_class=HTMLResponse)
async def convert_audio(
        request: Request,
):
    with open("audio_convert.html", "r", encoding="utf-8") as f:
        return f.read()


# 新增的 JavaScript 处理上传和播放音频
@tool_app.get("/audio/upload")
async def upload_audio_page(request: Request):
    html_content = """
    <!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音频转换上传</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
            font-size: 24px;
        }
        .container {
            max-width: 500px;
            margin: 0 auto;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .file-upload {
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            overflow: hidden;
            border: 2px dashed #ddd;
            border-radius: 5px;
            padding: 20px;
            cursor: pointer;
            transition: border-color 0.3s ease;
            background-color: #fafafa;
        }
        .file-upload:hover {
            border-color: #5cb85c;
        }
        .file-upload input[type="file"] {
            position: absolute;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }
        .file-upload-label {
            font-size: 18px;
            color: #555;
            font-weight: bold;
        }
        button {
            background-color: #5cb85c;
            color: white;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #4cae4c;
        }
        audio {
            margin-top: 20px;
            width: 100%;
            outline: none;
        }
        .file-name {
            margin-top: 10px;
            font-size: 16px;
            color: #333;
            text-align: center;
            font-weight: bold;
        }
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .input-group label {
            font-weight: bold;
            color: #555;
        }
        .input-group input {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>上传音频文件进行转换</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <label class="file-upload">
                <span class="file-upload-label">选择音频文件</span>
                <input type="file" id="audioFile" name="audio_file" accept="audio/*" required onchange="displayFileName()">
            </label>
            <div class="input-group">
                <label for="audioFormat">音频格式 (如 wav, mp3):</label>
                <input type="text" id="audioFormat" name="audio_format" placeholder="请输入音频格式" required>
            </div>
            <div class="input-group">
                <label for="audioSampleRate">采样率 (如 44100):</label>
                <input type="number" id="audioSampleRate" name="audio_sample_rate" placeholder="请输入采样率" required>
            </div>
            <button type="submit">上传并转换</button>
        </form>
        <div class="file-name" id="fileName"></div> <!-- 显示文件名的区域 -->
        <audio id="audioPlayer" controls style="display:none;"></audio>
    </div>

    <script>
        function displayFileName() {
            const audioFile = document.getElementById('audioFile').files[0]; // 获取选择的文件
            const fileNameDisplay = document.getElementById('fileName');

            if (audioFile) {
                fileNameDisplay.textContent = `已选择文件: ${audioFile.name}`; // 显示文件名
            } else {
                fileNameDisplay.textContent = ''; // 清空文件名
            }
        }

        document.getElementById('uploadForm').onsubmit = async function(event) {
            event.preventDefault(); // 阻止表单默认提交

            const formData = new FormData(this);
            const response = await fetch('/audio/convert', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const audioBlob = await response.blob(); // 获取音频的 Blob 对象
                const audioUrl = URL.createObjectURL(audioBlob); // 创建对象 URL
                const audioPlayer = document.getElementById('audioPlayer');
                audioPlayer.src = audioUrl; // 设置音频播放源
                audioPlayer.style.display = 'block'; // 显示音频播放器
                audioPlayer.play(); // 播放音频
            } else {
                console.error('音频转换失败:', response.statusText);
                alert('音频转换失败，请重试。');
            }
        };
    </script>
</body>
</html>

    """
    return HTMLResponse(content=html_content)


if __name__ == '__main__':
    video_info, openai_client = init_app()
    uvicorn.run(tool_app, host='0.0.0.0', port=8092)
    # uvicorn.run(tool_app, host='0.0.0.0', port=8090, workers=2, limit_concurrency=4, limit_max_requests=100)
