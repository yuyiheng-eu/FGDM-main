import os
import subprocess
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# === 参数配置 ===
openpose_bin_path = "/media/data_external/openpose/build/examples/openpose/openpose.bin"
openpose_model_path = "/media/data_external/openpose/models"
video_root_base = "/media/data_external/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
json_output_base_root = "/media/data_external/FGDM/preprocess/PHOENIX14T/openpose_output_json"

num_workers = 2 # 并发进程数
csv_files = glob.glob('*.csv')

def run_openpose(image_dir, output_dir):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) == len(os.listdir(image_dir)):
        return f"Skipped (already done): {image_dir}"

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        openpose_bin_path,
        "--image_dir", image_dir,
        "--write_json", output_dir,
        "--model_folder", openpose_model_path,
        "--display", "0",
        "--render_pose", "0",
        "--hand",
        "--number_people_max","1",
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Success: {image_dir}"
    except subprocess.CalledProcessError as e:
        return f"Failed: {image_dir}, error: {e}"

def process_csv_file(file_path):
    mode = file_path.split('.')[-3]  # e.g., dev.csv → mode=dev
    video_root = os.path.join(video_root_base, mode)
    json_output_base = os.path.join(json_output_base_root, mode)
    os.makedirs(json_output_base, exist_ok=True)

    df = pd.read_csv(file_path, sep='|', skiprows=1, header=None)

    tasks = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, row in df.iterrows():
            name = row.iloc[0]
            image_dir = os.path.join(video_root, name)
            output_dir = os.path.join(json_output_base, name)
            tasks.append(executor.submit(run_openpose, image_dir, output_dir))

        for i, future in enumerate(as_completed(tasks), 1):
            print(f"[{i}/{len(tasks)}] {future.result()}")

if __name__ == "__main__":
    for file in csv_files:
        print(f"\n Processing CSV: {file}")
        process_csv_file(file)
