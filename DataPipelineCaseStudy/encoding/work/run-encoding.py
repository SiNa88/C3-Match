import os
import sys
import time
from datetime import datetime
from subprocess import call
import subprocess

def print_call(call):
    s = ""
    for i in call:
        s += " " + str(i)
    print(s)

def main():

    if (len(sys.argv) != 2):
        print("Usage: run_encoder.py <codec>")
        print("Codecs: x264")
        sys.exit()
    codec_name = sys.argv[1]
    
    if (codec_name != "x264"):
        print("Usage: run_encoder.py <codec>")
        print("Codecs: x264")
        sys.exit()    
    
    out_extension = ".mp4"
    
    fps, file_pattern, idx_start, idx_end, orig_res = [24, "20{}", 30, 30, (4096, 1744)]
    
    #in_file_full = "~/Documents/NaMe/CODA/Traffic-sign-classification-microservices/0encoding-transcoding-packaging/20.mp4"
    in_file_name = "20.mp4"

    out_file = file_pattern.format(idx_start) + out_extension
    #print (out_file)
    #-----encoding------
    #quality_speed_options = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
    quality_speed = "ultrafast"
    
    enc_call = ["ffmpeg", "-y"]
    enc_call += ["-i", str(in_file_name)]
    enc_call += ["-r", str(fps)]

    bitrate, width, height = [20000, 3840, 2160]#(200, 320, 180),(1500, 1024, 576),(3000, 1280, 720),(6500, 2560, 1440),(12000, 3840, 2160),(20000, 3840, 2160)
    enc_call += ["-vf", "scale={}x{}".format(width, height), "-pix_fmt", "yuv420p"]

    enc_call += ["-c:v", "libx264", "-threads", "8"]

    #enc_call += ["-preset", "slow"]
    enc_call += ["-preset", "{}".format(quality_speed)]

    enc_call += ["-b", "{}k".format(bitrate)]
    enc_call += [out_file]
    
    print_call(enc_call)
    start_time = time.monotonic()
    call(enc_call)
    elapsed_time = time.monotonic() - start_time
    
    print("Encoding done - time:{}".format(elapsed_time))
    
if __name__ == "__main__":
    main()
