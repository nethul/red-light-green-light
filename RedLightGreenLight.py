import os
import sys
import time
import math
import random
import threading
import ctypes
import numpy as np
import cv2

from ultralytics import YOLO
from sort import Sort
from image_difference import Diff

# ----------------- Real-Time Audio Manager (Instant Cutoff Support) -----------------
class AudioManager:
    """
    Controls audio playback on Windows using winmm (MCI).
    Supports instant mid-vocal cutoff so players cannot predict when Red Light strikes.
    """
    def __init__(self):
        self.mci = ctypes.windll.winmm.mciSendStringW
        self.active_aliases = set()
        self.lock = threading.Lock()

    def play(self, sound_file, alias="sfx", loop=False):
        if not os.path.exists(sound_file):
            return
        path = os.path.abspath(sound_file)
        with self.lock:
            self._stop_alias(alias)
            self.mci(f'open "{path}" type mpegvideo alias {alias}', None, 0, 0)
            cmd = f'play {alias} from 0'
            if loop:
                cmd += ' repeat'
            self.mci(cmd, None, 0, 0)
            self.active_aliases.add(alias)

    def stop(self, alias="sfx"):
        with self.lock:
            self._stop_alias(alias)

    def _stop_alias(self, alias):
        self.mci(f'stop {alias}', None, 0, 0)
        self.mci(f'close {alias}', None, 0, 0)
        self.active_aliases.discard(alias)

    def stop_all(self):
        with self.lock:
            for a in list(self.active_aliases):
                self._stop_alias(a)

audio = AudioManager()


# ----------------- Pose & Hands-Up Helpers -----------------
def check_hands_up(kpt, conf=None):
    """
    Returns (hands_up: bool, left_up: bool, right_up: bool, status: str)
    A hand is up if the wrist is above the shoulder (y_wrist < y_shoulder).
    """
    if kpt is None or len(kpt) < 17:
        return False, False, False, "No Pose"

    # COCO Keypoints: 5=L Shoulder, 6=R Shoulder, 9=L Wrist, 10=R Wrist
    ls_x, ls_y = kpt[5]
    rs_x, rs_y = kpt[6]
    lw_x, lw_y = kpt[9]
    rw_x, rw_y = kpt[10]

    lw_c = conf[9] if conf is not None else 1.0
    rw_c = conf[10] if conf is not None else 1.0
    ls_c = conf[5] if conf is not None else 1.0
    rs_c = conf[6] if conf is not None else 1.0

    left_valid = (lw_c > 0.25 and ls_c > 0.25 and lw_y > 0 and ls_y > 0)
    right_valid = (rw_c > 0.25 and rs_c > 0.25 and rw_y > 0 and rs_y > 0)

    # In image coordinates, y is 0 at top, so wrist above shoulder means y_wrist < y_shoulder
    left_up = left_valid and (lw_y < ls_y)
    right_up = right_valid and (rw_y < rs_y)

    if left_valid and right_valid:
        is_up = (left_up and right_up)
        status = "Both Hands Up" if is_up else ("Left Hand Down" if not left_up else "Right Hand Down")
    elif left_valid:
        is_up = left_up
        status = "Hand Up" if is_up else "Hand Down"
    elif right_valid:
        is_up = right_up
        status = "Hand Up" if is_up else "Hand Down"
    else:
        is_up = False
        status = "Wrists Not In View"

    return is_up, left_up, right_up, status


def draw_arms_skeleton(img, kpt, conf=None):
    """Draw arm bones and wrist points on camera feed for real-time visual feedback."""
    if kpt is None or len(kpt) < 17:
        return

    connections = [
        (5, 7), (7, 9),   # Left arm: Shoulder -> Elbow -> Wrist
        (6, 8), (8, 10),  # Right arm: Shoulder -> Elbow -> Wrist
        (5, 6)            # Across Shoulders
    ]

    for p1, p2 in connections:
        c1 = conf[p1] if conf is not None else 1.0
        c2 = conf[p2] if conf is not None else 1.0
        if c1 > 0.25 and c2 > 0.25:
            pt1 = (int(kpt[p1][0]), int(kpt[p1][1]))
            pt2 = (int(kpt[p2][0]), int(kpt[p2][1]))
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(img, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

    # Draw wrist circles
    for p in [9, 10]:
        c = conf[p] if conf is not None else 1.0
        if c > 0.25:
            pt = (int(kpt[p][0]), int(kpt[p][1]))
            if pt[0] > 0 and pt[1] > 0:
                cv2.circle(img, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)


# ----------------- Motion Detection Helper -----------------
def check_player_moving(prev_box, crnt_box, prev_img, crnt_img,
                        bbox_thresh=7.0, img_diff_thresh=900, had_high_momentum=False):
    """
    Determine if an individually tracked player moved.
    If the player was moving fast when Red Light hit (high momentum), thresholds
    are slightly tighter to penalize physical stopping wobble/inertia.
    """
    if had_high_momentum:
        bbox_thresh = max(4.5, bbox_thresh * 0.75)
        img_diff_thresh = max(600, int(img_diff_thresh * 0.75))

    # 1. Centroid displacement
    prev_center = ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2)
    crnt_center = ((crnt_box[0] + crnt_box[2]) / 2, (crnt_box[1] + crnt_box[3]) / 2)
    dist = math.hypot(crnt_center[0] - prev_center[0], crnt_center[1] - prev_center[1])
    if dist >= bbox_thresh:
        prefix = "Inertia Overshoot: " if had_high_momentum else ""
        return True, f"{prefix}BBox shift ({dist:.1f}px)"

    # 2. Box dimension expansion/contraction
    w1 = abs(prev_box[2] - prev_box[0])
    h1 = abs(prev_box[3] - prev_box[1])
    w2 = abs(crnt_box[2] - crnt_box[0])
    h2 = abs(crnt_box[3] - crnt_box[1])
    if abs(w2 - w1) >= bbox_thresh or abs(h2 - h1) >= bbox_thresh:
        prefix = "Momentum Sway: " if had_high_momentum else ""
        return True, f"{prefix}BBox size change"

    # 3. Pixel differencing within player's bounding box
    if prev_img is not None and crnt_img is not None:
        diff = Diff(prev_img, crnt_img, threshold=25)
        ih, iw = diff.shape[:2]
        x1 = max(0, min(int(round(crnt_box[0])), iw - 1))
        y1 = max(0, min(int(round(crnt_box[1])), ih - 1))
        x2 = max(0, min(int(round(crnt_box[2])), iw))
        y2 = max(0, min(int(round(crnt_box[3])), ih))

        if x2 > x1 and y2 > y1:
            box_diff = diff[y1:y2, x1:x2]
            moved_pixels = int(np.sum(box_diff) / 255)
            if moved_pixels >= img_diff_thresh:
                prefix = "Stopping Wobble: " if had_high_momentum else ""
                return True, f"{prefix}In-box movement ({moved_pixels}px)"

    return False, ""


def draw_hud_box(img, x, y, w, h, bg_color=(0, 0, 0), alpha=0.6):
    """Draw a semi-transparent HUD background rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def match_box_to_detection(track_box, det_boxes):
    """Matches a SORT track box [x1, y1, x2, y2] to the closest YOLO detection box index."""
    best_idx = -1
    best_iou = 0.0
    for idx, dbox in enumerate(det_boxes):
        ix1 = max(track_box[0], dbox[0])
        iy1 = max(track_box[1], dbox[1])
        ix2 = min(track_box[2], dbox[2])
        iy2 = min(track_box[3], dbox[3])
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area1 = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
        area2 = (dbox[2] - dbox[0]) * (dbox[3] - dbox[1])
        union = area1 + area2 - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    if best_iou >= 0.2:
        return best_idx
    return -1


def open_camera():
    """Tries multiple camera indices and backends for Windows stability."""
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
    return None


def run_game():
    # ----------------- Load Graphic Frames -----------------
    CANVAS_W, CANVAS_H = 1280, 720
    
    frame_files = {
        'green': os.path.join('frames', '0.png'),
        'red': os.path.join('frames', '1.png'),
        'kill': os.path.join('frames', '2.png'),
        'winner': os.path.join('frames', '3.png'),
        'intro': os.path.join('frames', '4.png')
    }

    frames = {}
    for key, path in frame_files.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            frames[key] = cv2.resize(img, (CANVAS_W, CANVAS_H))
        else:
            print(f"Warning: {path} not found. Using blank placeholder.")
            frames[key] = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # ----------------- Sound File Paths -----------------
    sounds = {
        'intro': os.path.join('sounds', 'squidWin.mp3'),
        'chant': os.path.join('sounds', 'RLGLsong.mp3'),
        'gunshot': os.path.join('sounds', 'gunShot.mp3'),
        'player_out': os.path.join('sounds', 'playerOut.mp3'),
        'win': os.path.join('sounds', 'win.mp3'),
        'kill': os.path.join('sounds', 'kill.mp3')
    }

    # ----------------- Load YOLO Model & Tracker -----------------
    model_path = 'yolov8n-pose.pt'
    if not os.path.exists(model_path):
        model_path = os.path.join('neato_game', 'yolov8n-pose.pt')
    
    print(f"Loading YOLO tracking model from {model_path}...")
    model = YOLO(model_path)
    tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.25)
    print("Tracking model ready!")

    # ----------------- Initialize Camera -----------------
    cap = open_camera()
    if cap is None:
        print("Error: Could not open webcam.")
        return

    # Let camera warm up
    for _ in range(10):
        cap.read()

    window_name = 'Squid Game - Red Light, Green Light'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_W, CANVAS_H)

    # ----------------- Hands-Up Lobby (Hold 5s to Register as Player) -----------------
    audio.play(sounds['intro'], alias='intro_bgm', loop=True)
    REQUIRED_HOLD_SECONDS = 5.0

    registered_player_ids = set()
    hands_up_start_times = {}     # tid -> timestamp when hands went up
    auto_start_deadline = None    # Countdown timestamp to auto-start once players are ready
    AUTO_START_DELAY = 4.0

    LOBBY_CAM_W, LOBBY_CAM_H = 600, 420
    LOBBY_CAM_X = CANVAS_W - LOBBY_CAM_W - 40
    LOBBY_CAM_Y = 115

    print("Lobby active: Raise hands in sky for 5 seconds to register as a Player. Others will be Audience.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        lobby_canvas = frames['intro'].copy()

        # Run pose detection on webcam frame
        results = model(frame, imgsz=480, verbose=False)
        attr = results[0]
        boxes = attr.boxes.cpu().numpy()
        kpts_obj = attr.keypoints

        kpts_xy = kpts_obj.xy.cpu().numpy() if kpts_obj is not None else []
        kpts_conf = kpts_obj.conf.cpu().numpy() if (kpts_obj is not None and kpts_obj.conf is not None) else None

        xyxys = boxes.xyxy
        confs = boxes.conf

        if len(xyxys) > 0:
            dets = np.array([[xy[0], xy[1], xy[2], xy[3], confs[i]] for i, xy in enumerate(xyxys)])
            tracks = tracker.update(dets)
        else:
            tracks = tracker.update()

        cam_annotated = frame.copy()
        crnt_lobby_tracks = {}

        # Track active hands-up times and register players
        for track in tracks:
            tid = int(track[4])
            tbox = track[:4]
            crnt_lobby_tracks[tid] = tbox

            # Find matching YOLO detection
            d_idx = match_box_to_detection(tbox, xyxys)
            if d_idx != -1:
                kpt = kpts_xy[d_idx] if d_idx < len(kpts_xy) else None
                conf = kpts_conf[d_idx] if (kpts_conf is not None and d_idx < len(kpts_conf)) else None
                is_up, left_up, right_up, status_text = check_hands_up(kpt, conf)
            else:
                kpt, conf = None, None
                is_up, status_text = False, "Tracking..."

            x1, y1, x2, y2 = [int(v) for v in tbox]

            # Player registration logic
            if tid in registered_player_ids:
                # Already registered as Player!
                box_color = (0, 255, 0)
                tag = f"PLAYER #{tid} [REGISTERED]"
                draw_arms_skeleton(cam_annotated, kpt, conf)
                cv2.rectangle(cam_annotated, (x1, y1), (x2, y2), box_color, 3)

                tag_w = len(tag) * 11
                cv2.rectangle(cam_annotated, (x1, max(0, y1 - 25)), (x1 + tag_w, y1), box_color, -1)
                cv2.putText(cam_annotated, tag, (x1 + 4, max(15, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)

            elif is_up:
                # In process of registering!
                if tid not in hands_up_start_times:
                    hands_up_start_times[tid] = now
                held_sec = now - hands_up_start_times[tid]
                progress = min(1.0, held_sec / REQUIRED_HOLD_SECONDS)

                draw_arms_skeleton(cam_annotated, kpt, conf)

                if held_sec >= REQUIRED_HOLD_SECONDS:
                    registered_player_ids.add(tid)
                    hands_up_start_times.pop(tid, None)
                    print(f"Player #{tid} registered successfully!")
                    audio.play(sounds['win'], alias=f'reg_{tid}')
                    box_color = (0, 255, 0)
                    tag = f"PLAYER #{tid} [REGISTERED!]"
                else:
                    box_color = (0, 255, 255)  # Yellow
                    tag = f"REGISTERING #{tid}: {held_sec:.1f}s / 5.0s"

                cv2.rectangle(cam_annotated, (x1, y1), (x2, y2), box_color, 2)

                # Mini progress bar on bounding box
                pbar_w = max(40, x2 - x1)
                fill_w = int(pbar_w * progress)
                cv2.rectangle(cam_annotated, (x1, max(0, y1 - 32)), (x1 + pbar_w, max(0, y1 - 22)), (50, 50, 50), -1)
                cv2.rectangle(cam_annotated, (x1, max(0, y1 - 32)), (x1 + fill_w, max(0, y1 - 22)), (0, 255, 0), -1)

                tag_w = len(tag) * 11
                cv2.rectangle(cam_annotated, (x1, max(0, y1 - 20)), (x1 + tag_w, y1), box_color, -1)
                cv2.putText(cam_annotated, tag, (x1 + 4, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

            else:
                # Lowered hands or not attempting to register -> AUDIENCE
                hands_up_start_times.pop(tid, None)
                box_color = (150, 150, 150)  # Muted gray
                tag = f"Audience #{tid} [SPECTATOR]"
                cv2.rectangle(cam_annotated, (x1, y1), (x2, y2), box_color, 1)
                tag_w = len(tag) * 9
                cv2.rectangle(cam_annotated, (x1, max(0, y1 - 20)), (x1 + tag_w, y1), (60, 60, 60), -1)
                cv2.putText(cam_annotated, tag, (x1 + 4, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        # Starting condition evaluation
        registering_in_progress = len(hands_up_start_times) > 0
        start_game_now = False

        if len(registered_player_ids) >= 1:
            if registering_in_progress:
                # Pause auto-start countdown while someone is actively registering
                auto_start_deadline = None
                bottom_msg = "NEW PLAYER REGISTERING... HOLD HANDS IN SKY!"
                bar_color = (0, 200, 255)
                progress_val = 0.5
            else:
                if auto_start_deadline is None:
                    auto_start_deadline = now + AUTO_START_DELAY
                remaining_countdown = max(0.0, auto_start_deadline - now)
                bottom_msg = f"PLAYERS LOCKED! STARTING IN {remaining_countdown:.1f}s (or press [SPACE])"
                bar_color = (0, 220, 0)
                progress_val = 1.0 - (remaining_countdown / AUTO_START_DELAY)

                if remaining_countdown <= 0:
                    start_game_now = True
        else:
            auto_start_deadline = None
            if len(crnt_lobby_tracks) > 0:
                bottom_msg = "RAISE BOTH HANDS FOR 5s TO JOIN AS A PLAYER!"
                bar_color = (0, 140, 255)
                progress_val = 0.0
            else:
                bottom_msg = "STEP IN FRONT OF CAMERA TO JOIN"
                bar_color = (120, 120, 120)
                progress_val = 0.0

        # ----------------- Render Lobby HUD -----------------
        # Title Banner
        draw_hud_box(lobby_canvas, 40, 25, CANVAS_W - 80, 70, (10, 10, 10), 0.8)
        cv2.putText(lobby_canvas, "SQUID GAME - REGISTRATION LOBBY",
                    (60, 68), cv2.FONT_HERSHEY_DUPLEX, 1.05, (0, 255, 255), 2, cv2.LINE_AA)

        # Left Info Panel
        draw_hud_box(lobby_canvas, 40, 115, 540, 420, (15, 15, 15), 0.85)
        cv2.putText(lobby_canvas, "GAME ENTRY RULES:", (65, 150),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(lobby_canvas, "1. Raise hands in sky for 5 sec to JOIN as Player", (65, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(lobby_canvas, "2. Anyone else in view is AUDIENCE (spectators)", (65, 212),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(lobby_canvas, "3. Audience movement will NOT cause eliminations!", (65, 239),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1, cv2.LINE_AA)

        cv2.line(lobby_canvas, (65, 255), (550, 255), (60, 60, 60), 1)

        # Roster summary
        audience_count = len([tid for tid in crnt_lobby_tracks if tid not in registered_player_ids])
        cv2.putText(lobby_canvas, f"PLAYERS: {len(registered_player_ids)}   |   AUDIENCE: {audience_count}",
                    (65, 285), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Show list of registered players
        if len(registered_player_ids) > 0:
            for idx, pid in enumerate(sorted(list(registered_player_ids))[:4]):
                cv2.putText(lobby_canvas, f">> Player #{pid}: [READY TO PLAY]",
                            (75, 320 + idx * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(lobby_canvas, "No players registered yet. Raise hands to join!",
                        (75, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 200, 255), 2, cv2.LINE_AA)

        # Show who is currently registering
        if registering_in_progress:
            for tid, t_start in list(hands_up_start_times.items())[:2]:
                h_sec = now - t_start
                cv2.putText(lobby_canvas, f"Registering #{tid}: {h_sec:.1f}s / 5.0s",
                            (75, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # Controls footer
        cv2.putText(lobby_canvas, "Press [SPACE] to Force Start  |  [Q] to Quit",
                    (65, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

        # Live Camera Feed on Right
        cam_resized = cv2.resize(cam_annotated, (LOBBY_CAM_W, LOBBY_CAM_H))
        border_col = (0, 255, 0) if len(registered_player_ids) >= 1 else (0, 140, 255)
        cv2.rectangle(lobby_canvas, (LOBBY_CAM_X - 4, LOBBY_CAM_Y - 4),
                      (LOBBY_CAM_X + LOBBY_CAM_W + 4, LOBBY_CAM_Y + LOBBY_CAM_H + 4), border_col, 3)
        lobby_canvas[LOBBY_CAM_Y:LOBBY_CAM_Y + LOBBY_CAM_H, LOBBY_CAM_X:LOBBY_CAM_X + LOBBY_CAM_W] = cam_resized

        cv2.rectangle(lobby_canvas, (LOBBY_CAM_X, LOBBY_CAM_Y), (LOBBY_CAM_X + 260, LOBBY_CAM_Y + 28), (0, 0, 0), -1)
        cv2.putText(lobby_canvas, "CAMERA TRACKING & POSE", (LOBBY_CAM_X + 10, LOBBY_CAM_Y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, border_col, 1, cv2.LINE_AA)

        # ----------------- Bottom Status Bar -----------------
        BAR_X, BAR_Y = 40, 560
        BAR_W, BAR_H = CANVAS_W - 80, 65
        draw_hud_box(lobby_canvas, BAR_X, BAR_Y, BAR_W, BAR_H, (15, 15, 15), 0.9)

        if progress_val > 0.0:
            fill_w = int((BAR_W - 20) * max(0.0, min(1.0, progress_val)))
            cv2.rectangle(lobby_canvas, (BAR_X + 10, BAR_Y + 8), (BAR_X + 10 + fill_w, BAR_Y + BAR_H - 8), bar_color, -1)

        cv2.putText(lobby_canvas, bottom_msg, (BAR_X + 30, BAR_Y + 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Render lobby frame
        cv2.imshow(window_name, lobby_canvas)
        key = cv2.waitKey(20) & 0xFF

        # Check for auto-start or manual start
        if start_game_now:
            print(f"Starting game with {len(registered_player_ids)} registered players!")
            audio.stop('intro_bgm')
            for countdown in ["READY...", "SET...", "START!"]:
                cd_canvas = lobby_canvas.copy()
                draw_hud_box(cd_canvas, CANVAS_W // 2 - 250, CANVAS_H // 2 - 80, 500, 140, (0, 0, 0), 0.85)
                cv2.putText(cd_canvas, countdown, (CANVAS_W // 2 - 130, CANVAS_H // 2 + 15),
                            cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 255, 255), 4, cv2.LINE_AA)
                cv2.imshow(window_name, cd_canvas)
                cv2.waitKey(500)
            break

        if key in [ord(' '), 13]:  # Space or Enter force start
            if len(registered_player_ids) == 0:
                if len(crnt_lobby_tracks) > 0:
                    registered_player_ids = set(crnt_lobby_tracks.keys())
                else:
                    registered_player_ids.add(1)
            print(f"Manual start triggered! Registered players: {registered_player_ids}")
            audio.stop('intro_bgm')
            break
        elif key == ord('q'):
            audio.stop_all()
            cap.release()
            cv2.destroyAllWindows()
            return

    # ----------------- Phase Randomization Helpers -----------------
    def get_random_green_duration():
        """
        Picks random duration and pattern for Green Light:
        - 35% Snap Cutoff: Abrupt cutoff mid-chant (0.8s - 1.5s) to punish sprinters!
        - 45% Standard Sprint: Normal pace (2.0s - 3.2s)
        - 20% Deceptive Lull: Long window (3.8s - 4.8s) to bait false confidence
        """
        pat = random.choices(['snap', 'medium', 'long'], weights=[0.35, 0.45, 0.20])[0]
        if pat == 'snap':
            return random.uniform(0.8, 1.5), "SNAP CUTOFF [HIGH DANGER]"
        elif pat == 'medium':
            return random.uniform(2.0, 3.2), "STANDARD SPRINT"
        else:
            return random.uniform(3.8, 4.8), "DECEPTIVE LULL"

    def get_random_red_duration():
        """Randomized Red Light gaze: 2.0s to 3.8s."""
        return random.uniform(2.0, 3.8)

    # ----------------- Game State Variables -----------------
    GAME_DURATION = 45  # total game seconds
    game_start_time = time.time()

    is_green = True
    phase_duration, current_pattern_name = get_random_green_duration()
    phase_start_time = time.time()

    audio.play(sounds['chant'], alias='chant')

    prev_tracks = {}              # track_id -> [x1, y1, x2, y2]
    prev_frame = None

    player_velocities = {}        # track_id -> smoothed speed (px/sec)
    player_last_positions = {}    # track_id -> (cx, cy, timestamp)
    players_high_momentum = set() # track_ids moving fast right when Red Light struck

    eliminated_ids = set()
    latest_elimination_msg = ""
    elimination_msg_time = 0

    game_won = False
    game_over = False

    PIP_W, PIP_H = 340, 220
    PIP_X = CANVAS_W - PIP_W - 20
    PIP_Y = 20

    print(f"Game started! Registered players: {registered_player_ids} | Initial phase: {current_pattern_name} ({phase_duration:.1f}s)")

    # ----------------- Main Game Loop -----------------
    consecutive_dropped_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            consecutive_dropped_frames += 1
            if consecutive_dropped_frames > 30:
                print("Error: Lost camera stream.")
                break
            time.sleep(0.01)
            continue
        consecutive_dropped_frames = 0

        now = time.time()
        elapsed = now - game_start_time
        time_left = max(0, int(GAME_DURATION - elapsed))

        # Check for time out
        if time_left <= 0:
            game_over = True
            break

        # ----------------- Phase Switching with Instant Cutoffs -----------------
        if now - phase_start_time >= phase_duration:
            is_green = not is_green
            phase_start_time = now

            if is_green:
                phase_duration, current_pattern_name = get_random_green_duration()
                print(f"[GREEN LIGHT] Pattern: {current_pattern_name} (Duration: {phase_duration:.1f}s)")
                audio.play(sounds['chant'], alias='chant')
                players_high_momentum.clear()
            else:
                # RED LIGHT! ABRUPT CUTOFF!
                # Instantly cut off chant audio mid-sentence so players cannot predict the freeze!
                audio.stop('chant')
                phase_duration = get_random_red_duration()
                
                # Flag players moving with high momentum at the exact moment of the cutoff
                players_high_momentum = {tid for tid, spd in player_velocities.items() if spd > 140 and tid in registered_player_ids}
                if players_high_momentum:
                    print(f"[CUTOFF!] High momentum flagged on players: {players_high_momentum}")

        # ----------------- YOLO Detection & SORT Tracking -----------------
        results = model(frame, verbose=False)
        attr = results[0]
        boxes = attr.boxes.cpu().numpy()

        xyxys = boxes.xyxy
        confs = boxes.conf

        if len(xyxys) > 0:
            dets = np.array([[xy[0], xy[1], xy[2], xy[3], confs[i]] for i, xy in enumerate(xyxys)])
            tracks = tracker.update(dets)
        else:
            tracks = tracker.update()

        crnt_tracks = {}
        cam_annotated = frame.copy()

        # Update detected positions & track velocity ONLY for registered players
        for track in tracks:
            tid = int(track[4])
            box = track[:4]
            crnt_tracks[tid] = box

            if tid in registered_player_ids:
                # Velocity tracking for registered players (pixels/sec)
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                if tid in player_last_positions:
                    px, py, pt = player_last_positions[tid]
                    dt = max(0.005, now - pt)
                    inst_speed = math.hypot(cx - px, cy - py) / dt
                    smoothed = 0.65 * player_velocities.get(tid, inst_speed) + 0.35 * inst_speed
                    player_velocities[tid] = smoothed
                player_last_positions[tid] = (cx, cy, now)

        # ----------------- Red Light Motion Detection -----------------
        # IMPORTANT: ONLY check registered players! Audience members are completely ignored!
        if not is_green and prev_frame is not None:
            for tid, crnt_box in crnt_tracks.items():
                if tid not in registered_player_ids:
                    continue  # Audience: ignore completely! Never check or eliminate!

                if tid in eliminated_ids:
                    continue

                if tid in prev_tracks:
                    prev_box = prev_tracks[tid]
                    had_inertia = (tid in players_high_momentum)
                    moved, reason = check_player_moving(prev_box, crnt_box, prev_frame, frame,
                                                        had_high_momentum=had_inertia)
                    if moved:
                        eliminated_ids.add(tid)
                        latest_elimination_msg = f"PLAYER #{tid} ELIMINATED! ({reason})"
                        elimination_msg_time = now
                        print(f"[RED LIGHT] Player #{tid} moved ({reason}) -> ELIMINATED!")
                        
                        # Play gunshot sound immediately
                        audio.play(sounds['gunshot'], alias=f'shot_{random.randint(100, 999)}')
                        audio.play(sounds['player_out'], alias=f'out_{random.randint(100, 999)}')

        # Check if all registered players have been eliminated
        remaining_players = registered_player_ids - eliminated_ids
        if len(registered_player_ids) > 0 and len(remaining_players) == 0:
            print("All registered players have been eliminated!")
            game_over = True
            break

        # ----------------- Annotate Webcam Feed (Player vs Audience) -----------------
        for tid, box in crnt_tracks.items():
            x1, y1, x2, y2 = [int(v) for v in box]

            if tid in registered_player_ids:
                # Registered Player
                speed = player_velocities.get(tid, 0.0)

                if tid in eliminated_ids:
                    color = (0, 0, 255)  # Red
                    label = f"Player #{tid} [ELIMINATED]"
                elif is_green:
                    if speed > 220:
                        color = (0, 140, 255)  # Orange: High momentum risk!
                        label = f"Player #{tid} [FAST - HIGH MOMENTUM!]"
                    elif speed > 110:
                        color = (0, 255, 255)  # Yellow: Moving briskly
                        label = f"Player #{tid} [SPEED: {int(speed)} px/s]"
                    else:
                        color = (255, 200, 0)  # Cyan: Controlled
                        label = f"Player #{tid} [CONTROLLED]"
                else:
                    # Red Light state
                    if tid in players_high_momentum:
                        color = (0, 180, 255)  # Flashing amber warning
                        label = f"Player #{tid} [INERTIA SENSITIVE!]"
                    else:
                        color = (0, 255, 0)    # Green: Safe frozen
                        label = f"Player #{tid} [FROZEN]"

                cv2.rectangle(cam_annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(cam_annotated, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
            else:
                # Audience / Spectator (Not in game, movements ignored)
                color = (130, 130, 130)  # Gray
                label = f"Audience #{tid} [SPECTATOR]"
                cv2.rectangle(cam_annotated, (x1, y1), (x2, y2), color, 1)
                cv2.putText(cam_annotated, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        prev_tracks = crnt_tracks
        prev_frame = frame.copy()

        # ----------------- Render Main Canvas -----------------
        canvas = frames['green'].copy() if is_green else frames['red'].copy()

        # 1. Top Phase Banner
        if is_green:
            banner_bg = (0, 140, 0)
            banner_text = "GREEN LIGHT - MOVE FORWARD!"
            sub_text = f"Pattern: {current_pattern_name} | Audio can cut off at ANY millisecond!"
        else:
            banner_bg = (0, 0, 180)
            banner_text = "RED LIGHT - FREEZE! DO NOT MOVE!"
            sub_text = "Chant cut off abruptly! Beware of stopping inertia!"

        draw_hud_box(canvas, 30, 20, 680, 95, banner_bg, 0.8)
        cv2.putText(canvas, banner_text, (50, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, sub_text, (50, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 255, 220) if is_green else (200, 200, 255), 1, cv2.LINE_AA)

        # 2. Timer HUD
        timer_color = (0, 255, 0) if time_left > 15 else ((0, 200, 255) if time_left > 5 else (0, 0, 255))
        draw_hud_box(canvas, 30, 130, 250, 75, (20, 20, 20), 0.7)
        cv2.putText(canvas, "TIME REMAINING", (45, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{time_left:02d}s", (45, 195),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, timer_color, 2, cv2.LINE_AA)

        # 3. Player Scoreboard HUD
        audience_in_view = len([t for t in crnt_tracks if t not in registered_player_ids])
        draw_hud_box(canvas, 30, 220, 270, 125, (20, 20, 20), 0.7)
        cv2.putText(canvas, f"PLAYERS: {len(registered_player_ids)}   AUDIENCE: {audience_in_view}", (45, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"ALIVE: {len(remaining_players)}", (45, 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"ELIMINATED: {len(eliminated_ids)}", (45, 303),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "(Audience motion is ignored)", (45, 328),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        # 4. Elimination Alert Banner
        if now - elimination_msg_time < 3.0 and latest_elimination_msg:
            draw_hud_box(canvas, 200, CANVAS_H - 120, CANVAS_W - 400, 50, (0, 0, 200), 0.85)
            cv2.putText(canvas, latest_elimination_msg, (220, CANVAS_H - 85),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 5. Bottom Instructions
        draw_hud_box(canvas, 0, CANVAS_H - 45, CANVAS_W, 45, (10, 10, 10), 0.8)
        cv2.putText(canvas, "[W] Reach Camera & WIN  |  [R] Restart  |  [Q] Quit",
                    (CANVAS_W // 2 - 240, CANVAS_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # 6. Picture-in-Picture Webcam Overlay (Top Right)
        cam_pip = cv2.resize(cam_annotated, (PIP_W, PIP_H))
        
        # Border around PiP
        pip_border_color = (0, 255, 0) if is_green else (0, 0, 255)
        cv2.rectangle(canvas, (PIP_X - 3, PIP_Y - 3), (PIP_X + PIP_W + 3, PIP_Y + PIP_H + 3), pip_border_color, 3)
        canvas[PIP_Y:PIP_Y + PIP_H, PIP_X:PIP_X + PIP_W] = cam_pip
        
        # PiP Header Tag
        cv2.rectangle(canvas, (PIP_X, PIP_Y), (PIP_X + 160, PIP_Y + 22), (0, 0, 0), -1)
        cv2.putText(canvas, "LIVE TRACKING", (PIP_X + 8, PIP_Y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, pip_border_color, 1, cv2.LINE_AA)

        # ----------------- Render Frame & Check Keys -----------------
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF

        # Win condition: player reaches camera and hits 'w' during Green Light
        if key == ord('w'):
            if is_green:
                print("Player reached camera during Green Light! VICTORY!")
                game_won = True
                break
            else:
                print("Pressed W during Red Light! Disqualified!")
                game_over = True
                break

        elif key == ord('r'):
            print("Restarting game...")
            audio.stop_all()
            return run_game()

        elif key == ord('q'):
            print("Quitting game...")
            audio.stop_all()
            break

    cap.release()
    audio.stop('chant')

    # ----------------- Post-Game Screens -----------------
    if game_won:
        audio.stop_all()
        audio.play(sounds['win'], alias='win_sound')
        while True:
            win_canvas = frames['winner'].copy()
            draw_hud_box(win_canvas, 200, CANVAS_H - 120, CANVAS_W - 400, 80, (0, 100, 0), 0.8)
            cv2.putText(win_canvas, "VICTORY! YOU SURVIVED SQUID GAME!",
                        (240, CANVAS_H - 75), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(win_canvas, "Press [R] to Play Again  |  Press [Q] to Quit",
                        (360, CANVAS_H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, win_canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('r'):
                audio.stop_all()
                return run_game()
            elif key == ord('q'):
                audio.stop_all()
                break

    elif game_over:
        audio.stop_all()
        audio.play(sounds['kill'], alias='kill_sound')
        while True:
            kill_canvas = frames['kill'].copy()
            draw_hud_box(kill_canvas, 180, CANVAS_H - 120, CANVAS_W - 360, 80, (0, 0, 140), 0.8)
            cv2.putText(kill_canvas, "ALL PLAYERS ELIMINATED / TIME EXPIRED!",
                        (220, CANVAS_H - 75), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(kill_canvas, "Press [R] to Play Again  |  Press [Q] to Quit",
                        (360, CANVAS_H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, kill_canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('r'):
                audio.stop_all()
                return run_game()
            elif key == ord('q'):
                audio.stop_all()
                break

    audio.stop_all()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_game()
