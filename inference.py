import os.path
import pickle
from multiprocessing.dummy import Process, Manager, Queue
import cv2
import time
from options.hifi_test_options import HifiTestOptions
from face_feature.hifi_image_api import HifiImage

# close onnxruntime warning
import onnxruntime
onnxruntime.set_default_logger_severity(3)


class GenInput(Process):
    def __init__(self, feature_src_list_, frame_queue_in_, frame_queue_out_, video_cap, src_img_path):
        super().__init__()
        self.frame_queue_in = frame_queue_in_
        self.frame_queue_out = frame_queue_out_
        self.feature_src_list = feature_src_list_
        self.src_img_path = src_img_path
        self.video_cap = video_cap
        self.hi = HifiImage(crop_size=256)

    def run(self):
        src_latent, crop_face = self.hi.get_face_feature(self.src_img_path)
        human_feature = [src_latent, crop_face]
        self.feature_src_list.append([human_feature])

        count = index = 0
        while True:
            # import numpy as np
            # frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            have_frame, frame = self.video_cap.read()
            if not have_frame:
                self.frame_queue_in.put(None)
                print("no more frame")
                # video.release()
                break
            # print(frame.shape)
            self.frame_queue_in.put(frame)



def save_video_ffmpeg(video_path, swap_video_path, model_name=''):
    video_name = os.path.basename(video_path).split('.')[-2]
    audio_file_path = os.path.join(os.path.dirname(os.path.realpath(video_path)), video_name + '.wav')
    # audio_file_path = video_path.split('.')[-2] + '.wav'
    if not os.path.exists(audio_file_path):
        print('extract audio')
        os.system(
            'ffmpeg -y -hide_banner -loglevel error -i "'
            + str(video_path)
            + '" -f wav -vn  "'
            + str(audio_file_path)
            + '"'
        )
    else:
        print('audio file exist')
    if os.path.exists(audio_file_path):
        os.rename(swap_video_path, swap_video_path.replace('.mp4', '_no_audio.mp4'))
        print('add audio')
        # start = time.time()
        os.system(
            'ffmpeg -y -hide_banner -loglevel error  -i "'
            + str(swap_video_path.replace('.mp4', '_no_audio.mp4'))
            + '" -i "'
            + str(audio_file_path)
            # + '" -c:v copy "'
            + '" -c:v libx264 "'
            + '"-c:a aac -b:v 40000k "'
            + str(swap_video_path)
            + '"'
        )
        # print('add audio time cost', time.time() - start)
        # print('remove temp')
        os.remove(swap_video_path.replace('.mp4', '_no_audio.mp4'))
    if model_name != '':
        os.rename(swap_video_path, swap_video_path.replace('.mp4', '_%s.mp4' % model_name))
    os.remove(audio_file_path)

def chang_video_resolution(video_path, resize_video_path):
    print('change video resolution to 1080p')
    os.system(
        'ffmpeg -y -hide_banner -loglevel error -i "'
        + str(video_path)
        + '" -vf scale=1080:-1 -c:v libx264 -c:a aac -b:v 20000k "'
        + str(resize_video_path)
        + '"'
    )


class GetOutput(Process):
    def __init__(self, frame_queue_out_, src_video_path, model_name, out_dir, video_fps, video_size, video_frame_count, image_name,
                 align_method, use_gfpgan, sr_weight, use_color_trans=False, color_trans_mode='rct'):
    # def __init__(self, frame_queue_out_, src_video_path, model_name, out_dir, video_info):
        super().__init__()
        self.frame_queue_out = frame_queue_out_
        self.src_video_path = src_video_path
        out_video_name = image_name + '_to_' + os.path.basename(src_video_path).split('.')[-2] + '_' + model_name + '_' + align_method + '.mp4'
        if use_gfpgan:
            out_video_name = out_video_name.replace('.mp4', '_sr_{}.mp4'.format(sr_weight))
        if use_color_trans:
            out_video_name = out_video_name.replace('.mp4', '_'+color_trans_mode+'.mp4')
        self.out_path = os.path.join(out_dir, out_video_name)
        # self.video_info = video_info
        print(self.out_path)
        self.videoWriter = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, video_size)
        self.video_frame_count = video_frame_count
        # self.model_name = model_name



    def run(self):
        # import time
        count = 0
        fps_count = 0

        start_time = time.time()
        while True:
            queue_out = self.frame_queue_out.get()
            frame_out = queue_out
            # print("out:", type(queue_out))
            fps_count += 1

            if fps_count % 100 == 0:
                end_time = time.time()
                print('fps: {}'.format(fps_count / (end_time - start_time)))
                start_time = time.time()
                fps_count = 0
            count += 1
            if count % self.video_frame_count == 0:
                break
            self.videoWriter.write(frame_out)
        self.videoWriter.release()
        start_time = time.time()
        save_video_ffmpeg(self.src_video_path, self.out_path)
        print("add audio cost:", time.time() - start_time)



class FaceSwap(Process):
    def __init__(self, feature_src_list_, frame_queue_in_,
                 frame_queue_out_, model_name='', align_method='68', use_gfpgan=True, sr_weight=1.0,color_trans_mode='rct'):
        super().__init__()
        from HifiFaceAPI_parallel_trt_roi_realtime_sr_api import HifiFaceRealTime
        self.hfrt = HifiFaceRealTime(feature_src_list_, frame_queue_in_,
                                     frame_queue_out_, model_name=model_name, align_method=align_method,
                                     use_gfpgan=use_gfpgan, sr_weight=sr_weight, use_color_trans=False, color_trans_mode=color_trans_mode)
    def run(self):
        self.hfrt.forward()


if __name__ == '__main__':
    frame_queue_in = Queue(2)
    frame_queue_out = Queue(2)
    manager = Manager()
    image_feature_src_list = manager.list()
    opt = HifiTestOptions().parse()

    model_name = opt.model_name
    align_method = opt.align_method
    use_gfpgan = opt.use_gfpgan
    sr_weight = opt.sr_weight
    use_color_trans = opt.use_color_trans
    color_trans_mode = opt.color_trans_mode
    print("use_gfpgan:", use_gfpgan, "use use_color_trans:", use_color_trans)

    src_img_path = opt.src_img_path
    image_name = src_img_path.split('/')[-1].split('.')[0]
    video_path = opt.video_path
    print(video_path)
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = opt.output_dir
    output_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print("ori_video_size:", video_size)
    if video_size != (1080, 1920) and opt.video_to_1080p:
        resize_video_path = video_path.replace('.mp4', '_1080p.mp4')
        if not os.path.exists(resize_video_path):
            chang_video_resolution(video_path, resize_video_path)
        video_path = resize_video_path
        # video_size = (1080, 1920)

    t1 = time.time()
    gi = GenInput(image_feature_src_list, frame_queue_in, frame_queue_out, video, src_img_path)

    go = GetOutput(frame_queue_out, video_path, model_name, output_dir, video_fps, video_size, video_frame_count, image_name,
                   align_method, use_gfpgan, sr_weight, use_color_trans, color_trans_mode)

    fs = FaceSwap(image_feature_src_list, frame_queue_in, frame_queue_out,
                  model_name=model_name, align_method=align_method, use_gfpgan=use_gfpgan, sr_weight=sr_weight, color_trans_mode=color_trans_mode)

    gi.start()
    go.start()
    fs.start()

    gi.join()
    print('gi stop')
    go.join()
    print('go stop')
    fs.join()
    print('fs stop')

    video.release()

    print("time cost:", time.time()-t1)
