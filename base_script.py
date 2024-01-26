import torch
import torchvision
import torch
import time
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datasets import load_from_disk
from tqdm.autonotebook import tqdm

uri = "mongodb+srv://admin:admin@cluster0.3dy8ntv.mongodb.net/?retryWrites=true&w=majority"

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class ImageDataSet(torch.utils.data.Dataset):

    def __init__(self, pictures, list_of_idxs=[], transform_fn=lambda x: x):
        self.pictures = pictures
        self.list_of_idxs = list_of_idxs
        self.transform_fn = transform_fn
        self.current_idx = 0
        self.lenght = len(pictures)
        self.preprocessing_time = 0

    def transform(self, image):
        start = time.time()
        if image.mode == "L" or image.mode == "RGBA":
            image = image.convert('RGB')
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transformed = transforms(image)
        end = time.time()
        transform_time = end-start
        self.preprocessing_time += transform_time
        return transformed

    def __len__(self):
        return len(self.pictures)

    def __getitem__(self, idx):
        picture = self.transform(self.pictures[int(idx)]['image'])
        label = self.pictures[int(idx)]['label']
        self.current_idx = idx
        if len(self.pictures) - 1 <= idx:
            print("Preprocessing Time: " + str(self.preprocessing_time))
        return picture, label


class Box:
    def __init__(self, box, object_name, confidence):
        self.box = box
        self.xmin = box[0]
        self.ymin = box[1]
        self.xmax = box[2]
        self.ymax = box[3]
        self.object_name = object_name
        self.confidence = confidence

    def __str__(self):
        return f'Box({self.xmin},{self.ymin},{self.xmax},{self.ymax},{self.object_name},{self.confidence})'

    def __repr__(self):
        return self.__str__()


def transform_images(frame):
    return frame


def target_dnn_callback(target_dnn_output):
    boxes = target_dnn_output[0]['boxes'].detach().cpu().numpy()
    confidences = target_dnn_output[0]['scores'].detach().cpu().numpy()
    object_ids = target_dnn_output[0]['labels'].detach().cpu().numpy()
    label = []
    for i in range(len(boxes)):
        object_name = COCO_INSTANCE_CATEGORY_NAMES[object_ids[i]]
        if confidences[i] > 0.50 and object_name == 'car':
            box = Box(boxes[i], object_ids[i], confidences[i])
            label.append(box)
        label.append(None)
    return label


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)
model.eval()

dataset = load_from_disk("~/../../mount-ssd/hendrik/imageNet/train")
dataset = ImageDataSet(dataset, transform_fn=transform_images)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

boxes = {}


client = MongoClient(uri, server_api=ServerApi('1'))
db = client['ml-systems']
collection = db['maskrcnn_resnet50_fpn']

results = []

for data, label in tqdm(dataloader):
    eval = model(data)
    boxes = target_dnn_callback(eval)
    label = label.numpy()
    i = 0
    for box in boxes:
        if box is not None:
            results.append({str(label[i]): float(box.confidence)})
        i += 1

    if len(results) > 100:
        collection.insert_many(results)
        results = []

if len(results) > 0:
    collection.insert_many(results)
