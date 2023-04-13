import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from car_detect_yolo import yolo_filter_boxes, iou, yolo_eval, predict, yolo_non_max_suppression
from yolo_utils import read_classes, read_anchors,scale_boxes, preprocess_image

grades = {}

print("TESTING UNIT TEST 1")

tf.random.set_seed(10)
box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = 0.5)

try:
    assert type(scores) == EagerTensor, "Use tensorflow functions"
    assert type(boxes) == EagerTensor, "Use tensorflow functions"
    assert type(classes) == EagerTensor, "Use tensorflow functions"

    assert scores.shape == (1789,), "Wrong shape in scores"
    assert boxes.shape == (1789, 4), "Wrong shape in boxes"
    assert classes.shape == (1789,), "Wrong shape in classes"

    assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
    assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
    assert classes[2].numpy() == 8, "Values are wrong on classes"

    print("All tests passed!")
    grades[1] = "Pass"
except:
    print("FAILED UNIT TEST 1")
    grades[1] = "Fail"


# BEGIN UNIT TEST
print("TESTING UNIT TEST 2")
## Test case 1: boxes intersect
try:
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)
    print("iou for intersecting boxes = " + str(iou(box1, box2)))
    assert iou(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
    assert np.isclose(iou(box1, box2), 0.14285714), "Wrong value. Check your implementation. Problem with intersect."
    ## Test case 2: boxes do not intersect
    box1 = (1,2,3,4)
    box2 = (5,6,7,8)
    print("iou for non-intersecting boxes = " + str(iou(box1,box2)))
    assert iou(box1, box2) == 0, "Intersection must be 0"
    ## Test case 3: boxes intersect at vertices only
    box1 = (1,1,2,2)
    box2 = (2,2,3,3)
    print("iou for boxes that only touch at vertices = " + str(iou(box1,box2)))
    assert iou(box1, box2) == 0, "Intersection at vertices must be 0"
    ## Test case 4: boxes intersect at edge only
    box1 = (1,1,3,3)
    box2 = (2,3,3,4)
    print("iou for boxes that only touch at edges = " + str(iou(box1,box2)))
    assert iou(box1, box2) == 0, "Intersection at edges must be 0"

    print("All tests passed!")
    grades[2] = "Pass"
except:
    print("FAILED UNIT TEST 2")
    grades[2] = "Fail"
# END UNIT TEST

print("TESTING UNIT TEST 3")
try:
    # BEGIN UNIT TEST
    tf.random.set_seed(10)
    scores = tf.random.normal([54, ], mean=1, stddev=4, seed=1)
    boxes = tf.random.normal([54, 4], mean=1, stddev=4, seed=1)
    classes = tf.random.normal([54, ], mean=1, stddev=4, seed=1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    assert type(scores) == EagerTensor, "Use tensoflow functions"
    print("scores[2] = " + str(scores[2].numpy()))
    print("boxes[2] = " + str(boxes[2].numpy()))
    print("classes[2] = " + str(classes[2].numpy()))
    print("scores.shape = " + str(scores.numpy().shape))
    print("boxes.shape = " + str(boxes.numpy().shape))
    print("classes.shape = " + str(classes.numpy().shape))

    assert type(scores) == EagerTensor, "Use tensoflow functions"
    assert type(boxes) == EagerTensor, "Use tensoflow functions"
    assert type(classes) == EagerTensor, "Use tensoflow functions"

    assert scores.shape == (10,), "Wrong shape"
    assert boxes.shape == (10, 4), "Wrong shape"
    assert classes.shape == (10,), "Wrong shape"

    assert np.isclose(scores[2].numpy(), 8.147684), "Wrong value on scores"
    assert np.allclose(boxes[2].numpy(), [6.0797963, 3.743308, 1.3914018, -0.34089637]), "Wrong value on boxes"
    assert np.isclose(classes[2].numpy(), 1.7079165), "Wrong value on classes"

    print("All tests passed!")
    grades[3] = "Pass"
except Exception as e:
    print(e)
    print("FAILED UNIT TEST 3")
    grades[3] = "Fail"


# BEGIN UNIT TEST
print("TESTING UNIT TEST 4")
try:
    tf.random.set_seed(10)
    yolo_outputs = (tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
    tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
    tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
    tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].numpy()))
    print("boxes[2] = " + str(boxes[2].numpy()))
    print("classes[2] = " + str(classes[2].numpy()))
    print("scores.shape = " + str(scores.numpy().shape))
    print("boxes.shape = " + str(boxes.numpy().shape))
    print("classes.shape = " + str(classes.numpy().shape))
    assert type(scores) == EagerTensor, "Use tensoflow functions"
    assert type(boxes) == EagerTensor, "Use tensoflow functions"
    assert type(classes) == EagerTensor, "Use tensoflow functions"
    assert scores.shape == (10,), "Wrong shape"
    assert boxes.shape == (10, 4), "Wrong shape"
    assert classes.shape == (10,), "Wrong shape"
    assert np.isclose(scores[2].numpy(), 171.60194), "Wrong value on scores"
    assert np.allclose(boxes[2].numpy(), [-1240.3483, -3212.5881, -645.78, 2024.3052]), "Wrong value on boxes"
    assert np.isclose(classes[2].numpy(), 16), "Wrong value on classes"
    print("All tests passed!")
    grades[4] = "Pass"
except:
    print("FAILED UNIT TEST 4")
    grades[4] = "Fail"

# END UNIT TEST

# PREDICT
print("TESTING UNIT TEST 5")
try:
    out_scores, out_boxes, out_classes = predict("test.jpg")
    print("All tests passed!")
    grades[5] = "Pass"
except:
    print("FAILED UNIT TEST 5")
    grades[5] = "Fail"

print("FINAL GRADES:")
print(grades)
