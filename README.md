# Fight Analysis 🥊

**Fight Analysis** is a lightweight Python project that analyzes fight videos (UFC-style) using MediaPipe pose detection and simple heuristics. It detects strikes, takedowns, knockdowns, and assigns fighters by wrist tape color to compute scores and determine winners.

---
## Reason

- This project was built to help analyze fights that were considered **controversial, very close, or "robbery" decisions** by friends or viewers.
- It aims to provide an **objective way to track fighters’ movements, strikes, and actions** to better understand how scoring could have been impacted.
- The system can assist in identifying **key moments** like dominant strikes, head movement, or positioning that may have influenced judges’ decisions.
- This tool also serves as a **learning resource** for understanding fight dynamics, evaluating technique, and exploring how data can complement human observation.

## 🔧 Features

- Detects fighter poses using MediaPipe Pose Landmarker
- Assigns fighters by wrist tape color (red, blue)
- Detects actions: strikes, takedowns, knockdowns, head/body hits
- Keeps and displays scores in real-time
- Supports 2 or 3 fighters
- Auto-downloads the MediaPipe model if missing

---

## 🧰 Requirements

- Python 3.10+ recommended
- Libraries:
  - opencv-python
  - mediapipe
  - numpy

Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

> Note: On Windows, you may need to install a compatible version of Visual C++ Build Tools for some packages.

---

## 🚀 Usage

1. Place your video file in the project root and name it `video.mp4` (or modify `main.py` to point to another file).
2. Run the main script:

```bash
python main.py
```

3. When prompted, enter the number of fighters (2 or 3).
4. The app will open a window showing the video with scores and colored wrist indicators. Press `q` to quit.

The first run will download `pose_landmarker_lite.task` automatically if it is not present.

---

## 📂 Project Structure

- `main.py` — Entry point. Loads video, runs MediaPipe Landmarker, splits frame per fighter, detects actions, updates and displays scores.
- `color_tracker.py` — Detects dominant wrist tape color (red/blue/green), assigns colors to fighters, and draws indicators.
- `utils.py` — Action analyzers: `StrikeAnalyzer`, `TakedownAnalyzer`, `KnockdownAnalyzer`, `HeadHit`, `BodyHit`.
- `score.py` — `Fighter` class and helper functions `get_score`, `get_winner`.
- `pose_landmarker_lite.task` — MediaPipe pose model (auto-downloaded by `main.py`).

---

## 💡 Implementation Notes & Tips

- The color detection uses HSV thresholds and a small ROI around the wrist. If tape detection fails, try brighter/larger tape or adjust the HSV ranges in `color_tracker.py`.
- Heuristic thresholds (velocity, distance, cooldown frames) are set conservatively — tweak them in `utils.py` to match your camera/framerate.
- The video is split horizontally by the number of fighters; ensure fighters are roughly placed in each vertical slice.

---

## 🐛 Troubleshooting

- If no poses are detected, confirm the video resolution and lighting; MediaPipe performs best on clear footage.
- If model download fails, try downloading `pose_landmarker_lite.task` manually and place it in the project root.

---
## Example Fights

Here are some sample fights that can be analyzed with this project:

- **Justin Gaethje vs Dustin Poirier** – A close MMA match with striking and grappling exchanges.
- **Naoya Inoue vs Luis Nery** – A professional boxing match highlighting speed and precision.

## Limitations

- This project would benefit from an AI/ML model that can learn from fight data, as the scope of analyzing fights increases with understanding different fighting techniques.
- Certain actions, like feinting and head movement, could be considered scoring points at specific times, but are not currently detected.
- Not all MMA techniques are included due to the variety of striking and grappling disciplines.
- Camera angles and projections can confuse the detection system.
- Submissions and knockouts are not accounted for.

As a result, the system may have a **slightly higher chance of incorrectly predicting the winner by points**.

## 📜 License

Add your license here (e.g., MIT) or remove this section if not needed.

---

If you'd like, I can add a short example video or a sample config file to make setup even easier. ✅
