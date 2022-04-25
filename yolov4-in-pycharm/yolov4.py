import cv2 as cv
import time
import matplotlib.pyplot as plt
import cv2
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 0, 255), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('obj.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet("weights/custom.weights", "cfg/custom.cfg")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

#
# cap = cv.VideoCapture('output.avi')
starting_time = time.time()
frame_counter = 0
while True:
    frame = cv.imread('./live-images/17.jpg')
    frame_counter += 1
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid], score)
        cv.rectangle(frame, box, color, 3)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 1, color, 2)
        print(label)
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    cv.putText(frame, f'FPS: {fps}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    #cv.imshow('frame', frame)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.waitforbuttonpress()
    key = cv.waitKey(1)
    if key == ord('q'):
        break
# cap.release()
cv.destroyAllWindows()