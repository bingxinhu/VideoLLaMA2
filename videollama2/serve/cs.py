import spaces
import os
import re
import torch
import sys
sys.path.append('./')
import argparse
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init


class Chat:

    def __init__(self, model_path, load_8bit=False, load_4bit=False):
        disable_torch_init()

        self.model, self.processor, self.tokenizer = model_init(model_path, load_8bit=load_8bit, load_4bit=load_4bit)

    @spaces.GPU(duration=120)
    @torch.inference_mode()
    def generate(self, data: list, message, temperature, top_p, max_output_tokens):
        # TODO: support multiple turns of conversation.
        assert len(data) == 1

        tensor, modal = data[0]
        response = mm_infer(tensor, message, self.model, self.tokenizer, modal=modal.strip('<>'),
                            do_sample=True if temperature > 0.0 else False,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_output_tokens)

        return response


def process_inputs(image_path=None, video_path=None, text_message="", temperature=0.2, top_p=0.7, max_output_tokens=512):
    """
    处理输入数据，将图像、视频路径等转换为符合模型输入要求的数据格式
    """
    data = []
    processor = handler.processor
    try:
        if image_path is not None:
            data.append((processor['image'](image_path).to(handler.model.device, dtype=torch.float16), '<image>'))
        elif video_path is not None:
            data.append((processor['video'](video_path).to(handler.model.device, dtype=torch.float16), '<video>'))
        elif image_path is None and video_path is None:
            data.append((None, '<text>'))
        else:
            raise NotImplementedError("Not support image and video at the same time")
    except Exception as e:
        print(f"Error processing inputs: {e}")
        return []

    message = [{'role': 'user', 'content': text_message}]
    return data, message, temperature, top_p, max_output_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process inputs for the model.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the input image file.")
    parser.add_argument("--video_path", type=str, default=None, help="Path to the input video file.")
    parser.add_argument("--text_message", type=str, default="Describe what's in the image or video.",
                        help="Text message for the model input.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature parameter for generation.")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top P parameter for generation.")
    parser.add_argument("--max_output_tokens", type=int, default=512, help="Maximum number of output tokens.")
    args = parser.parse_args()

    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    handler = Chat(model_path, load_8bit=False, load_4bit=True)

    input_data, input_message, temp, top_p_value, max_tokens = process_inputs(
        image_path=args.image_path,
        video_path=args.video_path,
        text_message=args.text_message,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens
    )
    if input_data:
        result = handler.generate(
            input_data, input_message, temp, top_p_value, max_tokens
        )
        print("模型生成的回复:", result)
