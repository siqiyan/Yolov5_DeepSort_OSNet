#!/usr/bin/env python3
from tracker_lib import *
from transform_lib import *
import socket
import time
import argparse
MAXRECVLEN = 1024

class MsgCone:
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.confidence = 0.0
        self.ts = 0.0

    def to_bytes(self):
        s = '{:.6f},{:.6f},{:.6f},{}'.format(
            self.latitude,
            self.longitude,
            self.confidence,
            self.ts
        )
        return bytes(s, 'ascii')

class RTSP_URL:
    def __init__(self, ip, user, psw, channel='102'):
        self.ip = ip
        self.user = user
        self.psw = psw
        self.port = 554
        self.channel = channel
        self.prefix = 'rtspsrc location='
        self.opt = [
            'latency=100',
            'queue',
            'rtph264depay',
            'h264parse',
            'avdec_h264',
            'videoconvert',
            'videoscale',
            'video/x-raw,width=1280,height=720',
            'appsink'
        ]
        self.url = '{}rtsp://{}:{}@{}:{}/Streaming/Channels/{} {}'.format(
            self.prefix,
            self.user,
            self.psw,
            self.ip,
            self.port,
            self.channel,
            ' ! '.join(x for x in self.opt)
        )

    def get_url(self):
        return self.url

class WorkzoneServer:
    def __init__(self, args):
        self.args = args
        self.rsu_ip = [
            # '142.244.14.101' # 0
            # '192.168.10.1',
            '192.168.253.10',
        ]
        self.rsu_port = 4321
        self.init_rsu()

        self.cam_ip = [
            '142.244.14.93'
        ]
        streams_txt = '/tmp/streams.txt'
        with open(streams_txt, 'w') as f:
            if self.args.video_file:
                f.write(self.args.video_file)
            elif self.args.rtsp_user and self.args.rtsp_psw:
                for ip in self.cam_ip:
                    f.write(
                        '{}\n'.format(
                            RTSP_URL(
                                ip,
                                self.args.rtsp_user,
                                self.args.rtsp_psw
                            ).get_url()
                        )
                    )
            else:
                raise ValueError('Cannot determine proper video mode')

        self.packet_count = 0
        self.max_packet_count = 128

        self.tracker = Yolov5DeepSortTracker(
            streams_txt,
            self.proc_tracker_output
        )
        self.tracker.run()

    def init_rsu(self):
        self.rsu_socket = []
        for ip in self.rsu_ip:
            # Initialize socket:
            print('Trying to connect {}'.format(ip))
            self.rsu_socket.append(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            )
            self.rsu_socket[-1].connect((ip, self.rsu_port))
            print('Connect to {} success'.format(ip))

    def proc_tracker_output(self, outputs):
        # Process each input source
        # ts = time.time()
        ts = self.packet_count
        self.packet_count = (self.packet_count + 1) % self.max_packet_count
        for i in range(len(outputs)):
            if len(outputs[i]) > 0:
                # Process each detection
                for j, output in enumerate(outputs[i]):
                    lat, lon = det2gps(output)
                    msg = MsgCone()
                    msg.latitude = lat
                    msg.longitude = lon
                    msg.confidence = output[6]
                    msg.ts = ts
                    self.rsu_socket[i].send(msg.to_bytes())
                    recv_buf = self.rsu_socket[i].recv(MAXRECVLEN)
                    # msg = recv_buf.decode('utf-8')
                    # print('Received msg: {}'.format(msg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, required=False)
    parser.add_argument('--rtsp_user', type=str, required=False)
    parser.add_argument('--rtsp_psw', type=str, required=False)
    args = parser.parse_args()
    WorkzoneServer(args)
