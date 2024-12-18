import socket
import spaces
import os
import re
import torch
import sys
sys.path.append('./')
import argparse
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init


class ChatServer:
    def __init__(self, host, port, model_path, load_8bit=False, load_4bit=False):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"服务器正在监听 {self.host}:{self.port}...")

        disable_torch_init()
        self.model, self.processor, self.tokenizer = model_init(model_path, load_8bit=load_8bit, load_4bit=load_4bit)

    def handle_connection(self, client_socket):
        try:
            while True:
                # 接收客户端发送的数据
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                # 解析接收到的数据（这里假设客户端按固定格式发送，例如以逗号分隔的多个参数）
                parts = data.split(',')
                image_path = parts[0] if parts[0]!= "None" else None
                video_path = parts[1] if parts[1]!= "None" else None
                text_message = parts[2]
                temperature = float(parts[3])
                top_p = float(parts[4])
                max_output_tokens = int(parts[5])

                input_data, input_message, temp, top_p_value, max_tokens = self.process_inputs(
                    image_path=image_path,
                    video_path=video_path,
                    text_message=text_message,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens
                )
                if input_data:
                    result = self.generate(
                        input_data, input_message, temp, top_p_value, max_tokens
                    )
                    response = ",".join([str(r) for r in result])
                    client_socket.send(response.encode('utf-8'))
        except Exception as e:
            print(f"处理客户端连接出现错误: {e}")
        finally:
            client_socket.close()

    def process_inputs(self, image_path=None, video_path=None, text_message="", temperature=0.2, top_p=0.7, max_output_tokens=512):
        """
        处理输入数据，将图像、视频路径等转换为符合模型输入要求的数据格式
        """
        data = []
        processor = self.processor
        try:
            if image_path is not None:
                data.append((processor['image'](image_path).to(self.model.device, dtype=torch.float16), '<image>'))
            elif video_path is not None:
                data.append((processor['video'](video_path).to(self.model.device, dtype=torch.float16), '<video>'))
            elif image_path is None and video_path is None:
                data.append((None, '<text>'))
            else:
                raise NotImplementedError("Not support image and video at the same time")
        except Exception as e:
            print(f"Error processing inputs: {e}")
            return []

        message = [{'role': 'user', 'content': text_message}]
        return data, message, temperature, top_p, max_output_tokens

    @spaces.GPU(duration=120)
    @torch.inference_mode()
    def generate(self, data: list, message, temperature, top_p, max_output_tokens):
        """
        生成回复，现在支持多轮对话（对输入数据列表中的每一项依次处理）
        """
        responses = []
        for single_data in data:
            tensor, modal = single_data
            response = mm_infer(tensor, message, self.model, self.tokenizer, modal=modal.strip('<>'),
                                do_sample=True if temperature > 0.0 else False,
                                temperature=temperature,
                                top_p=top_p,
                                max_new_tokens=max_output_tokens)
            responses.append(response)
        return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoLLaMA2 Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器IP地址")
    parser.add_argument("--port", type=int, default=8888, help="服务器监听端口")
    parser.add_argument("--model_path", type=str, default='DAMO-NLP-SG/VideoLLaMA2.1-7B-16F', help="模型路径")
    args = parser.parse_args()

    server = ChatServer(args.host, args.port, args.model_path, load_8bit=False, load_4bit=True)
    while True:
        client_socket, client_address = server.server_socket.accept()
        print(f"客户端 {client_address} 已连接")
        server.handle_connection(client_socket)
