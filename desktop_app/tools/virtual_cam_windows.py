import argparse
import requests
import cv2
import numpy as np
import pyvirtualcam


def mjpeg_stream(url):
    resp = requests.get(url, stream=True)
    bytes_buffer = b""
    for chunk in resp.iter_content(chunk_size=1024):
        bytes_buffer += chunk
        a = bytes_buffer.find(b'\xff\xd8')
        b = bytes_buffer.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_buffer[a:b+2]
            bytes_buffer = bytes_buffer[b+2:]
            yield jpg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mjpeg', required=True, help='MJPEG URL from backend')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    with pyvirtualcam.Camera(width=args.width, height=args.height, fps=args.fps, print_fps=False) as cam:
        for jpg in mjpeg_stream(args.mjpeg):
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame = cv2.resize(frame, (args.width, args.height))
            cam.send(frame)
            cam.sleep_until_next_frame()


if __name__ == '__main__':
    main()