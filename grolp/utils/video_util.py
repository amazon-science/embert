import subprocess

import grolp.gen.constants as constants


class VideoSaver(object):

    def __init__(self, frame_rate=constants.VIDEO_FRAME_RATE):
        self.frame_rate = frame_rate

    def save(self, image_path, save_path):
        cmd = "ffmpeg -r %d -pattern_type glob -y -i '%s/*.png' -pix_fmt yuv420p '%s'" % (
            self.frame_rate, image_path, save_path)
        subprocess.call([cmd], shell=True)
