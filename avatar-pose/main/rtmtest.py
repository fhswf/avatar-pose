#import cv2
from pose_estimator_2d import PoseEstimator2D
from mmpose.models import build_posenet

estimator = PoseEstimator2D()

result = estimator.process_image("Gott1.jpeg")

# img = estimator.process_image_with_annotation("Gott1.jpeg", "Gott1-annotated.jpeg")

#save image
type(img)
#cv2.imwrite(, img)