import os
import subprocess
import glob
import pandas as pd

csv_files = glob.glob('*.csv')

for file in csv_files:

    mode = file.split('.')[-3]
    # === 配置部分 ===
    video_root = os.path.join("/media/data_external/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px",mode)
    openpose_bin_path = "/media/data_external/openpose/build/examples/openpose/openpose.bin"
    json_output_base = os.path.join("/media/data_external/FGDM/preprocess/PHOENIX14T/openpose_output_json",mode)  # JSON 输出根目录
    openpose_model_path =  "/media/data_external/openpose/models"

    # 确保输出根目录存在
    os.makedirs(json_output_base, exist_ok=True)



    df = pd.read_csv(file,sep='|',skiprows=1,header=None)

    for idx, row in df.iterrows():
        name = row.iloc[0]
        image_dir = os.path.join(video_root,name)
        output_dir = os.path.join(json_output_base, name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[{idx+1}/{df.shape[0]}] Mode:{mode} Processing: {image_dir}")

        cmd = [
            openpose_bin_path,
            "--image_dir", image_dir,
            "--write_json", output_dir,
            "--model_folder",openpose_model_path,
            "--display", "0",
            "--render_pose", "0",
            "--hand",
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"failed: {image_dir}\n error: {e}")

 

