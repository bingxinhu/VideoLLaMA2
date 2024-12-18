import socket


class ChatClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))

    def send_request(self, image_path, video_path, text_message, temperature, top_p, max_output_tokens):
        data = f"{image_path},{video_path},{text_message},{temperature},{top_p},{max_output_tokens}"
        self.client_socket.send(data.encode('utf-8'))
        response = self.client_socket.recv(4096).decode('utf-8')
        return response.split(',')

    def close_connection(self):
        self.client_socket.close()


if __name__ == "__main__":
    client = ChatClient("127.0.0.1", 8888)
    while True:
        image_path = input("请输入图像文件路径（没有则输入None）：")
        video_path = input("请输入视频文件路径（没有则输入None）：")
        text_message = input("请输入文本消息内容：")
        temperature = input("请输入温度参数（如0.2）：")
        top_p = input("请输入top_p参数（如0.7）：")
        max_output_tokens = input("请输入最大输出token数量（如512）：")

        result = client.send_request(
            image_path, video_path, text_message,
            float(temperature), float(top_p), int(max_output_tokens)
        )
        print("模型生成的回复:", result)

        choice = input("是否继续输入（y/n）：")
        if choice.lower()!= "y":
            client.close_connection()
            break
