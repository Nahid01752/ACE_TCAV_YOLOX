# ACE_TCAV_YOLOX
Adaptation of Automated Concept-based Explanation (ACE) XAI method to YOLOX object detection model

# Python libraries
- `torch`
- `torchvision`
- `scikit-image`
- `matplotlib`
- `tcav`

# Install YOLOX using its official GitHub repository:
- `git clone https://github.com/Megvii-BaseDetection/YOLOX.git`

# Download Pre-trained Model Weights

You can download the pre-trained YOLOX model weights on GTSRB dataset from Google Drive:

https://drive.google.com/file/d/1qWQzuP0ovgXvaCUDM6SH8tiiExMJbeY1/view?usp=drive_link

# Running the Code
python ace_run.py --num_parallel_workers 0 --target_class car --source_dir /COCO --working_dir /ACE_TCAV_YOLOX/outputs/test --model_to_run yolox-s --model_path /ACE_TCAV_YOLOX/yolox_s.pth --labels_path /ACE_TCAV_YOLOX/coco_classes.txt --feature_names backbone.C3_p4 backbone.C3_n3 backbone.C3_n4 --num_random_exp 50 --max_imgs 100 --min_imgs 30
