import cv2
import numpy as np
import pygame
from djitellopy import Tello
import cvzone

# --- PARAMETERS ---
thres = 0.55
nmsThres = 0.2
speed = 50  # viteză cm/s

# --- INIT CLASS NAMES AND MODEL ---
classNames = []
with open('coco.names', 'rt') as f:
    classNames = f.read().split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
#yolo v17
#dancenet inception net
weightsPath = "frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# --- INIT DRONE ---
pygame.init()
pygame.display.set_mode((200, 200))
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery(), "%")
tello.streamon()
frame_read = tello.get_frame_read()

running = True
flying = False

while running:
    # --- GET FRAME ---
    img = frame_read.frame
    if img is not None:
        img = cv2.resize(img, (640, 480))

        # --- OBJECT DETECTION ---
        classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
        try:
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cvzone.cornerRect(img, box)
                cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                            (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 255, 0), 2)
        except:
            pass

        # --- HUD ---
        battery = tello.get_battery()
        height = tello.get_height() if flying else 0
        yaw = tello.get_yaw() if flying else 0
        status = "In Aer" if flying else "La Sol"

        cv2.putText(img, f"Baterie: {battery}%", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Inaltime: {height} cm", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"Yaw: {yaw} deg", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img, f"Status: {status}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- ARROW ---
        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        arrow_y = center_y - 80
        arrow_length = 50
        cv2.line(img, (center_x, arrow_y + arrow_length), (center_x, arrow_y), (0, 0, 255), 3)
        pts = [(center_x - 10, arrow_y + 20),
               (center_x + 10, arrow_y + 20),
               (center_x, arrow_y)]
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (0, 0, 255))

        cv2.imshow("Tello + HUD + Detection", img)

    # --- EXIT KEY ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    # --- PYGAME INPUT ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    lr = 0
    fb = 0
    ud = 0
    yaw_vel = 0

    if keys[pygame.K_LEFT]:
        lr = -speed
    elif keys[pygame.K_RIGHT]:
        lr = speed
    if keys[pygame.K_UP]:
        fb = speed
    elif keys[pygame.K_DOWN]:
        fb = -speed
    if keys[pygame.K_w]:
        ud = speed
    elif keys[pygame.K_s]:
        ud = -speed
    if keys[pygame.K_a]:
        yaw_vel = -speed
    elif keys[pygame.K_d]:
        yaw_vel = speed

    # --- TAKEOFF / LAND / QUIT ---
    if keys[pygame.K_SPACE] and not flying:
        tello.takeoff()
        flying = True
    if keys[pygame.K_l] and flying:
        tello.land()
        flying = False
    if keys[pygame.K_q]:
        running = False

    # --- FLIPS ---
    if flying:
        if keys[pygame.K_f]:
            try: tello.flip_forward()
            except: pass
        if keys[pygame.K_b]:
            try: tello.flip_back()
            except: pass
        if keys[pygame.K_e]:
            try: tello.flip_left()
            except: pass
        if keys[pygame.K_r]:
            try: tello.flip_right()
            except: pass

    # --- SEND RC CONTROL ---
    if flying:
        tello.send_rc_control(lr, fb, ud, yaw_vel)

# --- CLEANUP ---
tello.send_rc_control(0, 0, 0, 0)
if flying:
    tello.land()
tello.streamoff()
tello.end()
cv2.destroyAllWindows()
pygame.quit()