import cv2
import numpy as np
import pygame
from djitellopy import Tello
import os
import time

# --- PARAMETERS --- 
thres = 0.55
nmsThres = 0.2
speed = 50  # viteză cm/s
BATTERY_UPDATE_INTERVAL = 2.0  # Actualizare baterie la fiecare 2 secunde
FLIP_COOLDOWN = 2.0  # Cooldown între flip-uri (secunde)

# --- TRACKING PARAMETERS ---
PERSON_CLASS_ID = 1  # ID-ul clasei "person" din COCO (index 0 + 1)
TRACKING_SPEED = 40  # Viteză de tracking (mai mică pentru mișcare mai smooth)
TRACKING_DEAD_ZONE = 50  # Zonă mortă în pixeli (evită oscilații)

# --- INIT CLASS NAMES AND MODEL ---
classNames = []
coco_names_path = 'coco.names'
if not os.path.exists(coco_names_path):
    raise FileNotFoundError(f"Fișierul {coco_names_path} nu a fost găsit!")
with open(coco_names_path, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

if not os.path.exists(configPath):
    raise FileNotFoundError(f"Fișierul {configPath} nu a fost găsit!")
if not os.path.exists(weightsPath):
    raise FileNotFoundError(f"Fișierul {weightsPath} nu a fost găsit!")

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# --- INIT DRONE ---
pygame.init()
pygame.display.set_mode((200, 200))
tello = Tello()

try:
    tello.connect()
    battery = tello.get_battery()
    print(f"Baterie: {battery}%")
    if battery < 10:
        print("⚠️  ATENȚIE: Bateria este foarte scăzută!")
    tello.streamon()
    frame_read = tello.get_frame_read()
except Exception as e:
    print(f"❌ Eroare la conectarea dronă: {e}")
    pygame.quit()
    exit(1)

running = True
flying = False
tracking_active = False  # Status tracking (toggle cu 't')

# --- TIMING VARIABLES ---
last_battery_update = 0
battery_cache = tello.get_battery()
last_flip_time = 0

# --- DISPLAY SIZE CONSTANTS ---
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# --- TRACKING VARIABLE ---
tracking_yaw_adjustment = 0  # Ajustare yaw pentru tracking

while running:
    # --- PYGAME INPUT - MUTAT ÎNAINTE PENTRU RESPONSIVITATE ---
    keys = pygame.key.get_pressed()
    lr = 0
    fb = 0
    ud = 0
    yaw_vel = 0

    # Procesare evenimente pentru toggle tracking și quit
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                tracking_active = not tracking_active
                if tracking_active:
                    print("🎯 Tracking activat - Dronă va urmări oamenii")
                else:
                    print("⏸️  Tracking dezactivat")
        if event.type == pygame.QUIT:
            running = False

    # Control manual cu tastele
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
    
    # --- TAKEOFF / LAND / QUIT - MUTAT AICI PENTRU RESPONSIVITATE ---
    if keys[pygame.K_SPACE] and not flying:
        try:
            print("🚁 Decolare...")
            tello.takeoff()
            flying = True
            print("✅ Dronă în aer!")
        except Exception as e:
            print(f"⚠️  Eroare la decolare: {e}")
            flying = False
    if keys[pygame.K_l] and flying:
        try:
            print("⬇️  Aterizare...")
            tello.land()
            flying = False
            print("✅ Dronă la sol!")
        except Exception as e:
            print(f"⚠️  Eroare la aterizare: {e}")
    if keys[pygame.K_q]:
        running = False

    # --- FLIPS - cu cooldown pentru a preveni apăsări accidentale multiple ---
    if flying:
        current_time = time.time()
        if current_time - last_flip_time >= FLIP_COOLDOWN:
            if keys[pygame.K_f]:
                try:
                    tello.flip_forward()
                    last_flip_time = current_time
                    print("🔄 Flip înainte executat")
                except Exception as e:
                    print(f"⚠️  Eroare la flip înainte: {e}")
            elif keys[pygame.K_b]:
                try:
                    tello.flip_back()
                    last_flip_time = current_time
                    print("🔄 Flip înapoi executat")
                except Exception as e:
                    print(f"⚠️  Eroare la flip înapoi: {e}")
            elif keys[pygame.K_e]:
                try:
                    tello.flip_left()
                    last_flip_time = current_time
                    print("🔄 Flip stânga executat")
                except Exception as e:
                    print(f"⚠️  Eroare la flip stânga: {e}")
            elif keys[pygame.K_r]:
                try:
                    tello.flip_right()
                    last_flip_time = current_time
                    print("🔄 Flip dreapta executat")
                except Exception as e:
                    print(f"⚠️  Eroare la flip dreapta: {e}")

    # --- GET FRAME ---
    img = frame_read.frame
    if img is not None:
        img = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # --- OBJECT DETECTION ---
        detected_person = None  # Bounding box pentru persoana detectată (cel mai mare/mare confidență)
        try:
            classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
            
            if len(classIds) > 0:
                classIds = classIds.flatten()
                confs = confs.flatten()
                
                # Căutare persoană pentru tracking (cea mai mare confidență)
                best_person_conf = 0
                
                for classId, conf, box in zip(classIds, confs, bbox):
                    # Validare classId pentru a evita index out of bounds
                    if 1 <= classId <= len(classNames):
                        className = classNames[classId - 1]
                        
                        # Verificare dacă este persoană și tracking activ
                        if classId == PERSON_CLASS_ID and tracking_active:
                            if conf > best_person_conf:
                                best_person_conf = conf
                                detected_person = box
                        
                        # Desenare bounding box pentru toate obiectele (x, y, w, h)
                        color = (0, 255, 255) if (classId == PERSON_CLASS_ID and tracking_active) else (0, 255, 0)
                        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, f'{className.upper()} {conf * 100:.2f}%',
                                    (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, color, 2)
        except Exception as e:
            print(f"⚠️  Eroare la detecție obiecte: {e}")

        # --- TRACKING PERSOANĂ ---
        tracking_yaw_adjustment = 0  # Resetare ajustare yaw
        if detected_person is not None and tracking_active and flying:
            # Calcul centru bounding box persoană
            person_center_x = detected_person[0] + detected_person[2] // 2
            person_center_y = detected_person[1] + detected_person[3] // 2
            
            # Centru cadru
            frame_center_x = DISPLAY_WIDTH // 2
            frame_center_y = DISPLAY_HEIGHT // 2
            
            # Calcul offset orizontal (doar pentru rotație stânga/dreapta)
            offset_x = person_center_x - frame_center_x
            
            # Dacă persoana nu este în dead zone, ajustează yaw
            if abs(offset_x) > TRACKING_DEAD_ZONE:
                # Normalizare offset și scalare la TRACKING_SPEED
                # Dacă persoana e în stânga, rotim stânga (yaw negativ)
                # Dacă persoana e în dreapta, rotim dreapta (yaw pozitiv)
                tracking_yaw_adjustment = int((offset_x / frame_center_x) * TRACKING_SPEED)
                # Limitare la TRACKING_SPEED
                tracking_yaw_adjustment = max(-TRACKING_SPEED, min(TRACKING_SPEED, tracking_yaw_adjustment))
            
            # Vizualizare tracking
            # Linie de la centru cadru la centru persoană
            cv2.line(img, (frame_center_x, frame_center_y), (person_center_x, person_center_y), 
                    (255, 0, 255), 2)
            # Cerc pe centru persoană
            cv2.circle(img, (person_center_x, person_center_y), 10, (255, 0, 255), -1)
            # Text tracking
            cv2.putText(img, "TRACKING", (person_center_x - 40, person_center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Aplicare tracking adjustment dacă tracking este activ și nu se controlează manual yaw
        if tracking_active and flying and not (keys[pygame.K_a] or keys[pygame.K_d]):
            yaw_vel = tracking_yaw_adjustment  # Tracking automat pentru persoană

        # --- HUD - OPTIMIZARE BATTERIE (cache cu timestamp) ---
        current_time = time.time()
        if current_time - last_battery_update >= BATTERY_UPDATE_INTERVAL:
            try:
                battery_cache = tello.get_battery()
                last_battery_update = current_time
            except Exception as e:
                print(f"⚠️  Eroare la citirea bateriei: {e}")
        
        battery = battery_cache
        try:
            height = tello.get_height() if flying else 0
            yaw = tello.get_yaw() if flying else 0
        except Exception as e:
            height = 0
            yaw = 0
            print(f"⚠️  Eroare la citirea senzorilor: {e}")
        
        # --- IMPROVED HUD DESIGN ---
        # Definire culori
        HUD_BG_ALPHA = 0.7  # Transparență fundal
        HUD_TEXT_COLOR = (255, 255, 255)  # Alb pentru text
        HUD_LABEL_COLOR = (200, 200, 200)  # Gri deschis pentru etichete
        
        # Panou stânga sus - Informații principale
        panel_x, panel_y = 10, 10
        panel_width, panel_height = 280, 220
        overlay = img.copy()
        
        # Fundal panou principal semi-transparent
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, HUD_BG_ALPHA, img, 1 - HUD_BG_ALPHA, 0, img)
        
        # Bordură panou
        cv2.rectangle(img, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (100, 100, 100), 2)
        
        # --- BATTERY ---
        battery_color = (0, 255, 0) if battery > 50 else (0, 165, 255) if battery > 20 else (0, 0, 255)
        battery_label_y = panel_y + 30
        
        # Icon baterie
        cv2.putText(img, "BAT", (panel_x + 10, battery_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_LABEL_COLOR, 1)
        
        # Bară de progres baterie
        bar_x, bar_y = panel_x + 60, battery_label_y - 15
        bar_width, bar_height = 200, 20
        bar_fill_width = int((battery / 100) * bar_width)
        
        # Fundal barei
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        # Bara de progres
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_fill_width, bar_y + bar_height),
                     battery_color, -1)
        # Contur barei
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (150, 150, 150), 2)
        # Text procentaj
        cv2.putText(img, f"{battery}%", (bar_x + bar_width + 10, battery_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, battery_color, 2)
        
        # --- HEIGHT ---
        height_label_y = battery_label_y + 40
        cv2.putText(img, "ALT", (panel_x + 10, height_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_LABEL_COLOR, 1)
        cv2.putText(img, f"{height:4d} cm", (panel_x + 60, height_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # --- YAW ---
        yaw_label_y = height_label_y + 40
        cv2.putText(img, "YAW", (panel_x + 10, yaw_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_LABEL_COLOR, 1)
        cv2.putText(img, f"{yaw:6.1f}", (panel_x + 60, yaw_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(img, "", (panel_x + 140, yaw_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_TEXT_COLOR, 1)
        
        # Compas vizual pentru yaw (mic)
        compass_x, compass_y = panel_x + 200, yaw_label_y - 20
        compass_radius = 20
        cv2.circle(img, (compass_x, compass_y), compass_radius, (100, 100, 100), 2)
        # Linie indicator yaw
        yaw_rad = np.radians(yaw)
        end_x = int(compass_x + compass_radius * 0.8 * np.cos(yaw_rad))
        end_y = int(compass_y - compass_radius * 0.8 * np.sin(yaw_rad))  # - pentru coordonate imagini
        cv2.line(img, (compass_x, compass_y), (end_x, end_y), (255, 100, 100), 2)
        # Nord indicator
        cv2.line(img, (compass_x, compass_y - compass_radius - 3), 
                (compass_x, compass_y - compass_radius + 3), (255, 255, 255), 2)
        
        # --- STATUS ---
        status_label_y = yaw_label_y + 40
        status_text = "IN FLIGHT" if flying else "GROUNDED"
        status_color = (0, 255, 0) if flying else (128, 128, 128)
        cv2.putText(img, "STAT", (panel_x + 10, status_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_LABEL_COLOR, 1)
        cv2.putText(img, status_text, (panel_x + 60, status_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # --- TRACKING STATUS ---
        tracking_label_y = status_label_y + 40
        tracking_status = "ACTIVE" if tracking_active else "OFF"
        tracking_color = (0, 255, 0) if tracking_active else (128, 128, 128)
        cv2.putText(img, "TRCK", (panel_x + 10, tracking_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_LABEL_COLOR, 1)
        cv2.putText(img, tracking_status, (panel_x + 60, tracking_label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_color, 2)
        
        # Panou dreapta sus - Compas central și crosshair
        center_x, center_y = DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2
        
        # Crosshair îmbunătățit
        crosshair_size = 30
        crosshair_thickness = 2
        crosshair_color = (0, 255, 255)  # Cyan
        
        # Linii orizontale și verticale
        cv2.line(img, (center_x - crosshair_size, center_y), 
                (center_x - crosshair_size//3, center_y), 
                crosshair_color, crosshair_thickness)
        cv2.line(img, (center_x + crosshair_size//3, center_y), 
                (center_x + crosshair_size, center_y), 
                crosshair_color, crosshair_thickness)
        cv2.line(img, (center_x, center_y - crosshair_size), 
                (center_x, center_y - crosshair_size//3), 
                crosshair_color, crosshair_thickness)
        cv2.line(img, (center_x, center_y + crosshair_size//3), 
                (center_x, center_y + crosshair_size), 
                crosshair_color, crosshair_thickness)
        
        # Cerc central mic
        cv2.circle(img, (center_x, center_y), 5, crosshair_color, -1)
        cv2.circle(img, (center_x, center_y), 8, crosshair_color, 1)
        
        # Săgeată sus (indicator direcție înainte)
        arrow_y = center_y - 60
        arrow_length = 40
        arrow_head_size = 8
        cv2.line(img, (center_x, arrow_y + arrow_length), (center_x, arrow_y), 
                (0, 0, 255), 2)
        pts = np.array([(center_x - arrow_head_size, arrow_y + arrow_head_size),
                       (center_x + arrow_head_size, arrow_y + arrow_head_size),
                       (center_x, arrow_y)], np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 255))
        
        # Text "N" (Nord) deasupra săgeții
        cv2.putText(img, "N", (center_x - 5, arrow_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Tello + HUD + Detection", img)

    # --- SEND RC CONTROL - MUTAT AICI PENTRU RESPONSIVITATE MAXIMĂ ---
    # Trimite comenzi chiar dacă nu există frame nou, pentru responsivitate maximă
    if flying:
        tello.send_rc_control(lr, fb, ud, yaw_vel)

    # --- EXIT KEY ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# --- CLEANUP ---
print("\n🛑 Oprire program...")
try:
    tello.send_rc_control(0, 0, 0, 0)
    if flying:
        print("⬇️  Aterizare...")
        tello.land()
        flying = False
    tello.streamoff()
    tello.end()
except Exception as e:
    print(f"⚠️  Eroare la cleanup: {e}")
finally:
    cv2.destroyAllWindows()
    pygame.quit()
    print("✅ Program terminat corect.")