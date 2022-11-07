import colorsys
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

from module.centernet_resnet import center_net_resnet18, center_net_resnet34, center_net_resnet50
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image, show_config
from utils.utils_bbox import decode_bbox, postprocess
from utils.draw_picture import draw_heatmap, draw_heatmap_without_image


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class PersonNet(object):
    def __init__(self, **kwargs):
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        self.weights_path = r''
        self.classes_path = r'../model_data/person.txt'

        # 用于选择所使用的模型的主干：resnet50, hourglass, resnet34, resnet18
        self.backbone = 'resnet18'

        # 输入图片的大小，设置成32的倍数，（高，宽）
        self.input_shape = [288, 512]

        # 只有得分大于置信度的预测框会被保留下来
        self.confidence = 0.3

        # 非极大抑制所用到的nms_iou大小
        self.nms_iou = 0.3

        # --------------------------------------------------------------------------#
        #   是否进行非极大抑制，可以根据检测效果自行选择
        #   backbone为resnet50时建议设置为True、backbone为hourglass时建议设置为False
        # --------------------------------------------------------------------------#
        self.nms = True

        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        self.letterbox_image = False

        # 是否使用Cuda
        self.cuda = True if torch.cuda.is_available() else False

        # 计算总的类的数量
        self.class_names, self.num_classes = get_classes(self.classes_path)

        self.net = None
        self.generate()

    # 载入模型
    def generate(self, onnx=False):
        # 载入模型与权值
        assert self.backbone in ['resnet18', 'resnet34', 'resnet50']
        if self.backbone == "resnet18":
            self.net = center_net_resnet18(num_classes=self.num_classes, pretrained=False)
        elif self.backbone == "resnet34":
            self.net = center_net_resnet34(num_classes=self.num_classes, pretrained=False)
        elif self.backbone == "resnet50":
            self.net = center_net_resnet50(num_classes=self.num_classes, pretrained=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.weights_path != '':
            model_weights = torch.load(self.weights_path, map_location=device)
            self.net.load_state_dict(model_weights)
            print('{} model, and classes loaded.'.format(self.weights_path))

        self.net = self.net.eval()
        if not onnx:
            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # 检测图片，针对输入图像的尺度，给出该尺度下人的位置
    def detect_image(self, image: Image.Image):
        # 计算输入图片的高和宽
        # image_shape = np.array(np.shape(image)[0:2])
        image_shape = (image.height, image.width)
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        # 图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            # 将图像输入网络当中进行预测！
            outputs = self.net(images)

            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]

            # 利用预测结果进行解码
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            # -------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            # -------------------------------------------------------#
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            # 如果没有检测到物体，则返回原图
            if results[0] is None:
                return None

            top_label = results[0][:, 5]
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        # 进行目标的裁剪
        for i, c in list(enumerate(top_label)):
            top, left, bottom, right = top_boxes[i]
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            top_boxes[i] = top, left, bottom, right

        return top_label

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))


if __name__ == '__main__':
    from PIL import Image

    centernet = PersonNet()
    img = r"C:\Users\18457\PycharmProjects\centernet-pytorch-source\VOCdevkit/VOC2007/JPEGImages/1.png"
    image = Image.open(img)
    r_image = centernet.detect_image(image)
