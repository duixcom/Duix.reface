# -- coding: utf-8 --
# @Time : 2022/11/8


from cv2box import CVImage, MyFpsCounter
from model_lib import ModelBase
import numpy as np
import cv2

MODEL_ZOO = {
    # https://github.com/xuanandsix/GFPGAN-onnxruntime-demo
    # input_name:['input'], shape:[[1, 3, 512, 512]]
    # output_name:['1392'], shape:[[1, 3, 512, 512]]
    'GFPGANv1.4': {
        'model_path': './pretrain_models/gfpganv14_fp32_bs1_scale.onnx'
    },
    'codeformer': {
        'model_path':'./pretrain_models/codeformer_fp32_bs1_scale_adain.onnx'
    },

}


class GFPGAN(ModelBase):
    def __init__(self, model_type='GFPGANv1.4', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.input_std = self.input_mean = 127.5
        self.input_size = (512, 512)
        self.model_type = model_type

    def forward(self, face_image):
        """
        Args:
            face_image: cv2 image -1~1 RGB
        Returns:
            RGB 256x256x3 -1~1
        """
        face_image = (face_image + 1) / 2
        face_image_h, face_image_w, _ = face_image.shape
        if face_image_h != 512:
            face_image = cv2.resize(face_image, (512, 512))

        face_image = np.uint8(face_image * 255.0)
        image_in = CVImage(face_image).blob(self.input_size, self.input_mean, self.input_std, rgb=False)
        # image_in = CVImage(face_image).set_blob(self.input_std, self.input_mean, self.input_size).blob_in(rgb=False)
        if 'codeformer' in self.model_type:
            image_out = self.model.forward([image_in,np.array(1,dtype=np.float32)])
        else:
            image_out = self.model.forward(image_in)

        # print(image_out[0][0].shape)
        output_face = ((image_out[0][0] + 1) / 2).transpose(1, 2, 0).clip(0, 1)
        if face_image_h != 512:
            output_face = cv2.resize(output_face, (face_image_w, face_image_h))
        output_face = (output_face * 2 - 1.0)
        return output_face


if __name__ == '__main__':
    face_img_p = 'data/source/ym-1.jpeg'
    fa = GFPGAN(model_type='GFPGANv1.4', provider='gpu')
    with MyFpsCounter() as mfc:
        for i in range(10):
            face = fa.forward(face_img_p)
    # CVImage(face, image_format='cv2').save('./gfpgan.jpg')
    CVImage(face, image_format='cv2').show()
