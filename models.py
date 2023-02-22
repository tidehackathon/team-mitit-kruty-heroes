import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as album


def get_inference_augmentation():
    transform = [
        album.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        ),
    ]
    return album.Compose(transform)


class SegModel:
    NCLS = 1

    def __init__(self, weights_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._init_model()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_inference_augmentation()

    def infer_single_image(self, image):
        if self.transform:
            image = self.transform(image=image)['image']
        image = np.expand_dims(np.transpose(image, (2,0,1)), 0)
        image_tensor = torch.tensor(image).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self.model(image_tensor)
        output = self._generate_output(pred[0])
        return output

    def _init_model(self):
        raise NotImplementedError

    def _generate_output(self):
        raise NotImplementedError


class UnetRoad(SegModel):
    NCLS = 1

    def _init_model(self):
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=self.NCLS,
        )
        self.model = model

    def _generate_output(self, pred):
        pred = pred.detach().sigmoid().cpu().numpy()[0]
        return (pred*255).astype(np.uint8)


class UnetFour(SegModel):
    NCLS = 5

    def _init_model(self):
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=self.NCLS,
        )
        self.model = model

    def _generate_output(self, pred):
        pred = torch.argmax(pred.detach(), dim=0).cpu().numpy()
        return pred


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import cv2
    model_cls = UnetRoad('weights/uav_road_segm_efn0.pt')
    model_cls4 = UnetFour('weights/uav_4cl_segm_efn0.pt')

    image = cv2.cvtColor(cv2.imread('DSC06202.JPG'), cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, ((image.shape[1]//32)*32, (image.shape[0]//32)*32))
    image = cv2.resize(image, (1920, 1280))
    print(image.shape)

    mask0 = model_cls.infer_single_image(image)
    mask1 = model_cls4.infer_single_image(image)
    print(image.shape, mask0.shape)
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.imshow(mask0)
    plt.subplot(1,3,3)
    plt.imshow(mask1.astype(np.float32))
    plt.show()
