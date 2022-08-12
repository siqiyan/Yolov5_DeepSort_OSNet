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
