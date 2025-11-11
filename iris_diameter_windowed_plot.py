
import argparse
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# MediaPipe iris indices (requires refine_landmarks=True)
LEFT_IRIS_IDX = list(range(468, 473))
RIGHT_IRIS_IDX = list(range(473, 478))

def landmarks_to_pixels(landmarks, w, h):
    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks], dtype=np.int32)

def iris_diameter_from_points(pts):
    if pts is None or len(pts) == 0:
        return np.nan, (np.nan, np.nan)
    (x, y), r = cv2.minEnclosingCircle(pts.astype(np.float32))
    return 2.0 * r, (int(x), int(y))

def extract_frame_diameters(video_path, max_frames=None, display=False):
    """Process video and return DataFrame with per-frame diameters and timestamps."""
    if video_path.isdigit() or video_path == "0":
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = []
    pbar = tqdm(total=total_frames if total_frames else None, desc="Frames", unit="f")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_face_mesh.process(rgb)

            left_diam = np.nan
            right_diam = np.nan

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                left_pts = landmarks_to_pixels([lm[i] for i in LEFT_IRIS_IDX], w, h)
                right_pts = landmarks_to_pixels([lm[i] for i in RIGHT_IRIS_IDX], w, h)
                left_diam, _ = iris_diameter_from_points(left_pts)
                right_diam, _ = iris_diameter_from_points(right_pts)

                # optional display
                if display:
                    for p in left_pts:
                        cv2.circle(frame, tuple(p), 1, (0,255,0), -1)
                    for p in right_pts:
                        cv2.circle(frame, tuple(p), 1, (0,255,0), -1)
                    if not np.isnan(left_diam):
                        cv2.putText(frame, f"L:{left_diam:.1f}px", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    if not np.isnan(right_diam):
                        cv2.putText(frame, f"R:{right_diam:.1f}px", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            timestamp = frame_idx / fps
            rows.append({"frame": frame_idx, "time_s": timestamp, "left_px": left_diam, "right_px": right_diam})
            if display:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            pbar.update(1)
            if max_frames and frame_idx >= max_frames:
                break
    finally:
        pbar.close()
        cap.release()
        mp_face_mesh.close()
        if display:
            cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    # compute mean eye diameter (ignores NaNs)
    df["mean_px"] = df[["left_px", "right_px"]].mean(axis=1)
    return df, fps

def window_aggregate(df, window_s=2.5):
    """Aggregate frame-level diameters into windows of length window_s seconds."""
    # create time bin id
    df = df.copy()
    df["window_id"] = (df["time_s"] // window_s).astype(int)
    agg = df.groupby("window_id").agg(
        start_time_s=("time_s", "min"),
        end_time_s=("time_s", "max"),
        left_mean_px=("left_px", "mean"),
        right_mean_px=("right_px", "mean"),
        mean_px=("mean_px", "mean"),
        samples=("mean_px", "count"),
    ).reset_index(drop=False)
    # center time for plotting
    agg["time_center_s"] = (agg["start_time_s"] + agg["end_time_s"]) / 2.0
    return agg

def read_questions_csv(qpath):
    """Read questions CSV if provided. Expects columns start_s,end_s,label,difficulty (difficulty optional)."""
    qdf = pd.read_csv(qpath)
    # make sure columns exist
    required = {"start_s","end_s"}
    if not required.issubset(set(qdf.columns)):
        raise ValueError("Questions CSV must have columns 'start_s' and 'end_s' (in seconds). Optionally: 'label','difficulty'.")
    # normalize difficulty values if present
    if "difficulty" in qdf.columns:
        def map_diff(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (int,float)):
                return float(v)
            s = str(v).strip().lower()
            if s.startswith("e"): return 1.0
            if s.startswith("m"): return 2.0
            if s.startswith("d"): return 3.0
            try:
                return float(s)
            except:
                return np.nan
        qdf["difficulty_val"] = qdf["difficulty"].apply(map_diff)
    else:
        qdf["difficulty_val"] = np.nan
    return qdf

def generate_default_questions(total_duration_s, question_duration_s=10.0):
    """Partition the video into sequential question intervals of length question_duration_s."""
    qlist = []
    n = int(np.ceil(total_duration_s / question_duration_s))
    for i in range(n):
        start = i * question_duration_s
        end = min((i + 1) * question_duration_s, total_duration_s)
        qlist.append({"start_s": start, "end_s": end, "label": f"Q{i+1}", "difficulty_val": np.nan})
    return pd.DataFrame(qlist)

def overlay_load_on_plot(ax, qdf, norm_factor=None, alpha=0.15):
    """Shade question intervals and return a secondary axis with normalized expected load (1-3)."""
    # map difficulty_val to 1-3 already handled; if missing, skip
    for _, row in qdf.iterrows():
        s = row["start_s"]
        e = row["end_s"]
        ax.axvspan(s, e, color="gray", alpha=alpha)
        lbl = row.get("label", "")
        if lbl and hasattr(lbl, "__str__"):
            ax.text((s+e)/2.0, ax.get_ylim()[1]*0.98, str(lbl), ha="center", va="top", fontsize=8, alpha=0.8)

    # if difficulty values present, plot on secondary axis
    if qdf["difficulty_val"].notna().any():
        sec = ax.twinx()
        # build a step trace of expected load
        times = []
        vals = []
        for _, row in qdf.iterrows():
            times.extend([row["start_s"], row["end_s"]])
            vals.extend([row["difficulty_val"], row["difficulty_val"]])
        # normalize to same vertical scale as mean_px for visualization or use separate scale
        # Here we plot as 1-3 on secondary y-axis
        sec.step(times, vals, where="post", linewidth=1.5, label="Expected load (1=easy,3=hard)", color="orange")
        sec.set_ylabel("Expected load (1-3)")
        sec.set_ylim(0, 4)
        return sec
    return None

def plot_windowed(agg_df, questions_df=None, title="Iris Diameter (windowed)"):
    fig, ax = plt.subplots(figsize=(11,5))
    # plot mean (window centers)
    ax.plot(agg_df["time_center_s"], agg_df["mean_px"], marker="o", linestyle="-", linewidth=1.5, label="Measured mean iris diameter (px)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean iris diameter (px)")
    ax.set_title(title)
    ax.grid(True)
    # overlay question intervals & expected load if available
    sec = None
    if questions_df is not None and len(questions_df)>0:
        sec = overlay_load_on_plot(ax, questions_df)
    ax.legend(loc="upper left")
    if sec:
        sec.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", required=True, help="Path to video file (or '0' for webcam)")
    parser.add_argument("--window", "-w", type=float, default=2.5, help="Aggregation window in seconds (e.g., 2.5)")
    parser.add_argument("--questions", "-q", default=None, help="Optional questions CSV with start_s,end_s,label,difficulty")
    parser.add_argument("--display", action="store_true", help="Show frame display while processing")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick test")
    parser.add_argument("--question-duration", type=float, default=10.0, help="If no questions CSV, default question length (s) to split video")
    args = parser.parse_args()

    video_path = args.video
    if not (video_path.isdigit() or os.path.exists(video_path)):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("Extracting per-frame diameters...")
    df_frames, fps = extract_frame_diameters(video_path, max_frames=args.max_frames, display=args.display)
    total_duration = df_frames["time_s"].max() if not df_frames["time_s"].isna().all() else 0.0
    print(f"Video FPS: {fps:.2f}, duration ~ {total_duration:.2f}s, frames: {len(df_frames)}")

    print(f"Aggregating into windows of {args.window} seconds...")
    agg = window_aggregate(df_frames, window_s=args.window)

    qdf = None
    if args.questions:
        if not os.path.exists(args.questions):
            raise FileNotFoundError(f"Questions CSV not found: {args.questions}")
        qdf = read_questions_csv(args.questions)
    else:
        qdf = generate_default_questions(total_duration, question_duration_s=args.question_duration)

    # save aggregated results
    out_csv = "iris_windowed_aggregates.csv"
    agg.to_csv(out_csv, index=False)
    print(f"Saved windowed aggregates: {out_csv}")

    # Plot
    plot_windowed(agg, questions_df=qdf, title=f"Iris diameter (window {args.window}s)")

if __name__ == "__main__":
    main()