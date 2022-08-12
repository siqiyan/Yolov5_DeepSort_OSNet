#!/usr/bin/env python3
from rtsp_url import RTSP_URL
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp_ip', type=str, required=True)
    parser.add_argument('--rtsp_user', type=str, required=True)
    parser.add_argument('--rtsp_psw', type=str, required=True)
    args = parser.parse_args()
    url = RTSP_URL(
        args.rtsp_ip,
        args.rtsp_user,
        args.rtsp_psw
    ).get_url()
    cam = cv2.VideoCapture(url)
    state = 'streaming'
    writer = None
    video_name = None
    while True:
        ret, frame = cam.read()
        if state == 'streaming':
            cv2.putText(
                frame,
                'Streaming',
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        elif state == 'recording':
            writer.write(frame)
            cv2.putText(
                frame,
                'Recording',
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )
        else:
            raise ValueError('Unknown state: {}'.format(state))

        cv2.imshow('rtsp: {}'.format(args.rtsp_ip), frame)
        key = cv2.waitKey(5) & 0xff
        if key == ord('q'):
            break
        elif key == ord('r'):
            if state == 'streaming':
                video_name = '{}.mp4'.format(int(time.time()))
                writer = cv2.VideoWriter(
                    video_name,
                    cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                    10, # recording fps needs to match the camera fps
                    (1280, 720) # resolution needs to match camera resolution
                )
                state = 'recording'
                print('Start recording {}'.format(video_name))
        elif key == ord('s'):
            if state == 'recording':
                writer.release()
                writer = None
                state = 'streaming'
                print('Video has been saved to {}'.format(video_name))

    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print('Video has been saved to {}'.format(video_name))
