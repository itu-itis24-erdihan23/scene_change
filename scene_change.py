import cv2
import numpy as np
import os

def svd_custom(A, tol=1e-10, max_iter=100):
    m, n = A.shape
    ATA = A.T @ A
    eigvals, V = np.linalg.eig(ATA)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    singular_values = np.sqrt(np.abs(eigvals))
    Sigma = np.diag(singular_values)

    U = A @ V
    for i in range(V.shape[1]):
        norm = np.linalg.norm(U[:, i])
        if norm > 1e-8:
            U[:, i] /= norm

    return U, Sigma, V.T

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
    cap.release()
    return frames

def compute_tad(f1, f2):
    return np.sum(np.abs(f1 - f2))

def detect_shot_boundaries(video_path, output_dir, tad_thresh=100e6, svd_thresh=60000):
    frames = extract_frames(video_path)
    candidate_indices = []
    tad_values = []

    for i in range(len(frames) - 1):
        tad = compute_tad(frames[i], frames[i + 1])
        tad_values.append(tad)
        if tad > tad_thresh:
            candidate_indices.append(i)

    detected_transitions = []
    for idx in candidate_indices:
        if idx + 1 >= len(frames):
            continue

        f1 = frames[idx]
        f2 = frames[idx + 1]

        A1 = f1.reshape(-1, 1)
        A2 = f2.reshape(-1, 1)

        _, S1, _ = svd_custom(A1)
        _, S2, _ = svd_custom(A2)

        s1 = np.diag(S1)
        s2 = np.diag(S2)

        min_len = min(len(s1), len(s2))
        dist = np.linalg.norm(s1[:min_len] - s2[:min_len])

        if dist > svd_thresh:
            detected_transitions.append(idx)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "transitions.txt"), "w") as f:
        f.write("Detected Shot Transitions:\n")
        for idx in detected_transitions:
            f.write(f"Frame {idx}\n")
            cv2.imwrite(os.path.join(output_dir, f"transition_{idx}.jpg"),
                        frames[idx].astype(np.uint8))

if __name__ == "__main__":
    video_file = r"C:\Users\cumal\OneDrive\Masaüstü\github\scene_change\testvideo.mp4"
    output_folder = "results"
    detect_shot_boundaries(video_file, output_folder)