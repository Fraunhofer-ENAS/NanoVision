# This script is adapted from the official detectron2 tutorial for instance segmentation:
# TODO: Refactor it and turn it into a notebook!
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
import sys
setup_logger()

# Import common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image  # For handling TIFF images

# Import Detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from multiprocessing import freeze_support
import subprocess
# Add the project root to sys.path
# Note an alternative approach is to comment this line out and just run the 
# script from the terminal using -m flag as follows:
# python -m Experiments.Modified_Stardist.detectron_baseline_models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.append(project_root)

from Evaluation.Metrics_utils import *




image_path = r'.\Data\detectron2\images\train\400-1605-W1_map_00040_3.jpg'

# Load the TIFF image using PIL and convert to OpenCV format
print("Loading image...")
im = np.array(Image.open(image_path))  # Load with PIL
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

# Check if the image was loaded successfully
if im is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")
else:
    print("Image loaded successfully!")

# Display the input image
print("Displaying input image...")
cv2.imshow("Input Image", im)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()


register_coco_instances(
    "my_dataset_train",  # Dataset name
    {},  # Metadata (can be empty)
    r"Data\detectron2\annotations\train\coco_annotations.json",  # Path to annotations
    r"Data\detectron2\images\train"  # Path to images
)

# Register your validation dataset
register_coco_instances(
    "my_dataset_val",  # Dataset name
    {},  # Metadata (can be empty)
    r"Data\detectron2\annotations\val\coco_annotations.json",  # Path to annotations
    r"Data\detectron2\images\val"  # Path to images
   )

# Function to visualize a random sample from the dataset
def visualize_dataset(dataset_name, num_samples=5):
    # Get the dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)
    print(f"Number of samples in {dataset_name}: {len(dataset_dicts)}")

    metadata = MetadataCatalog.get(dataset_name)

    # Randomly sample images from the dataset
    for d in random.sample(dataset_dicts, num_samples):
        # Load the image
        a = d["file_name"]
        img = cv2.imread(d["file_name"])

        # Visualize the annotations
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)

        # Display the image with annotations
        cv2.imshow(f"Sample from {dataset_name}", vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

# Visualize the training dataset
# visualize_dataset("my_dataset_train", num_samples=5)

# Visualize the validation dataset
# visualize_dataset("my_dataset_val", num_samples=5)


# from detectron notebook
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.INPUT.MIN_SIZE_TRAIN = (256,640, 672, 704, 736, 768, 800)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 5  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10  # Run only 10 iterations (very short test) 300 iterations seems good enough for their balloon dataset
cfg.TEST.EVAL_PERIOD = 0  # Disable evaluation during training (faster)
cfg.SOLVER.STEPS = []  # No learning rate decay
cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Avoid frequent checkpoint savingcfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.DEVICE = "cpu"

"""os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()"""

subprocess.Popen(["tensorboard", "--logdir", "output"])


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
model_weights = r"Experiments\Benchmark\Detectron2\detectron2_models\detectron2_epochs1000_steps100__lr0.0001_lossmae"
cfg.MODEL.WEIGHTS = os.path.join(model_weights, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode
cfg.DATASETS.TEST = ("my_dataset_val",)
dataset_dicts = DatasetCatalog.get("my_dataset_val")

metadata = MetadataCatalog.get("my_dataset_val")
pred, gt,imgs = [], [], []
index = 0
# dataset_dicts = get_balloon_dicts("balloon/val")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=None,
                   scale=2,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    a = out.get_image()[:, :, ::-1]
    cv2.imshow("Input Image", out.get_image()[:, :, ::-1])
    cv2.imwrite(f"./Global_Outputs/Detectron_baseline_test/detectron_pred_{index}.png", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
    index += 1
    # setting up variables for evaluation
    imgs.append(im)
    pred.append(outputs)
    gt.append(d)


dst_dir = "./Global_Outputs/Detectron_baseline_test"

main_evaluation_loop(imgs, pred, dst_dir,  gt)


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("my_dataset_val", output_dir=dst_dir)
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
