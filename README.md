#  RED LIGHT GREEN LIGHT
A version of the iconic game as seen in the Netflix series: Squid Game, created using Open CV.
# Red Light, Green Light (Squid Game Edition)

## OBJECTIVE
Travelling towards the PC camera, the aim is to finally hit the 'w' key to win.
An advanced real-time computer vision implementation of the iconic **Red Light, Green Light** game from the Netflix series *Squid Game*. This project merges interactive arcade gameplay and visual UI with robust multi-person pose estimation and tracking (powered by **YOLOv8 Pose**, **SORT**, and pixel-level motion differencing).

## RULES
- You move during the red light period, you lose.
- You don't hit the 'w' key before the counter goes to zero, you lose.
- You hit 'q' key in between the game, you lose.
---

The code uses a simple motion detection technique to detect movements from the instant the doll opens its eyes (red light). 
It can be played with multiple players, too. Anyone moves during red light, the game gets over.
## 🎮 Key Features

Find demo here: https://youtu.be/WMmskDZWDMI 
1. **Player vs. Audience Registration (Lobby Phase)**:
   - Anyone standing in front of the camera is detected with pose estimation and assigned a SORT tracking ID.
   - To register as an active **Player**, hold both hands in the air for **5 continuous seconds**.
   - Anyone who does not hold hands up remains classified as **Audience** (spectators).
   - **Audience motion is completely ignored** during Red Light checks — spectators can walk around, cheer, or move without triggering gunshots or eliminations.

2. **Unpredictable Audio & Mid-Sentence Cutoffs**:
   - Built with Windows MCI low-latency audio control (`AudioManager`).
   - The eerie chant ("*Mugunghwa kkochi pieot seumnida*") does **not** always play to completion.
   - Dynamic tempo patterns randomly trigger:
     - **Snap Cutoff (0.8s – 1.5s)**: Abrupt early freeze to catch sprinters off-guard.
     - **Standard Sprint (2.0s – 3.2s)**: Normal paced window.
     - **Deceptive Lull (3.8s – 4.8s)**: Extended green light that baits players into reckless high speeds.

3. **Momentum & Stopping Inertia Penalization**:
   - Real-time player velocity (px/sec) is computed using smoothed centroid tracking.
   - If a player was sprinting fast at the exact millisecond Red Light struck, tight inertia thresholds are applied to penalize physical stopping wobble and momentum overshoot.

4. **Authentic Squid Game UI & HUD**:
   - Authentic full-screen Squid Game backdrops and sound effects (chant, gunshots, elimination announcements, victory, defeat).
   - Live Picture-in-Picture (PiP) tracking feed displaying bounding boxes, arm skeletons, speed gauges, and player states (`[CONTROLLED]`, `[HIGH MOMENTUM]`, `[FROZEN]`, `[ELIMINATED]`, and `[SPECTATOR]`).
   - Real-time scoreboard with alive/eliminated counts and countdown timer.

---

## 📁 Repository Structure

```
├── RedLightGreenLight.py    # Main Squid Game application
├── sort.py                  # SORT tracking (Kalman Filter + Hungarian matching)
├── image_difference.py      # Pixel differencing for in-box micro-motion
├── yolov8n-pose.pt          # YOLOv8 Pose neural network model
├── frames/                  # Squid Game UI visual assets (0-4 png)
├── sounds/                  # Audio cues (chants, gunshots, win, kill)
├── requirements.txt         # Unified Python dependencies
├── neato_game/              # Merged Neato robotics & vision package
│   ├── README.md            # Neato game documentation & analysis
│   ├── package.xml          # ROS2 package manifest
│   ├── setup.py             # Package setup
│   ├── img/                 # Research and visualization graphics
│   └── neato_game/          # Helper modules, vision tests, and utilities
```

---

## 🚀 Setup & Installation

### 1. Requirements
- Python 3.10+ (tested on Python 3.13)
- Webcam

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Game
```bash
python RedLightGreenLight.py
```

---

## 🕹️ Controls
- **Raise Hands for 5s**: Register as Player during Lobby
- **[SPACE]**: Force Start game from Lobby
- **[W]**: Reach the camera during Green Light to **WIN**
- **[R]**: Restart game
- **[Q]**: Quit
