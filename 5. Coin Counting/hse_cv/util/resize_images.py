import os
import cv2
import imghdr
import shutil


class ImageResizer:
    def __init__(self, input_dir, scaling_factor, output_dir='out'):
        self.input_dir = input_dir
        self.scaling_factor = scaling_factor
        self.output_dir = output_dir
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    def recursive_resize_images(self, relative_path=''):
        cur_dir = os.path.join(self.input_dir, relative_path)
        if not os.path.exists(os.path.join(self.output_dir, relative_path)):
            os.makedirs(os.path.join(self.output_dir, relative_path))
        for f in os.listdir(cur_dir):
            input_path = os.path.join(cur_dir, f)
            if os.path.isdir(input_path):
                self.recursive_resize_images(os.path.join(relative_path, f))
            elif imghdr.what(input_path) == 'jpeg':
                img = cv2.imread(input_path)
                out_img = cv2.resize(img, None, fx=self.scaling_factor, fy=self.scaling_factor,
                                     interpolation=cv2.INTER_LINEAR)
                output_path = os.path.join(self.output_dir, relative_path, f)
                cv2.imwrite(output_path, out_img)


def resize_images(input_dir, scaling_factor, output_dir):
    if output_dir is None:
        output_dir = 'resized_images'
    ir = ImageResizer(input_dir, scaling_factor, output_dir)
    ir.recursive_resize_images()
    print('Finish resizing images.')
