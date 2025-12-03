from ultralytics import YOLO


def main():
    # ---------------------------------------------------------
    # 1. SETUP: Load the Medium model
    # ---------------------------------------------------------
    # It will auto-download 'yolo11m-seg.pt' if you don't have it
    model = YOLO("yolo11m-seg.pt")

    # ---------------------------------------------------------
    # 2. TRAINING: The "Heavy Lifting"
    # ---------------------------------------------------------
    results = model.train(
        # --- PATHS ---
        # Update this to point to your specific yaml file
        data="datasets/pig-segmentation-HIT/data.yaml",

        # --- OUTPUT ---
        project="pig_project",  # Folder where results save
        name="pig_checkerboard_v1",  # Name of this specific run

        # --- HARDWARE (RTX 3090) ---
        device=0,  # Use your RTX 3090
        batch=0.85,  # Use 85% of VRAM
        workers=8,  # Use your CPU to load data

        # --- SETTINGS ---
        epochs=100,  # 100 loops
        patience=15,  # Stop early if no improvement for 15 epochs
        imgsz=640,  # Image size

        # --- AUGMENTATIONS (Top-Down & Checkerboard Tuned) ---
        degrees=180,  # Full rotation (pigs can face any way)
        flipud=0.5,  # Flip upside down (valid for top-down)
        fliplr=0.5,  # Flip left-right
        mosaic=1.0,  # Handle crowding
        mixup=0.15,  # Slight transparency mixing (helps separation)
        copy_paste=0.1,  # Pastes pigs onto different backgrounds

        # --- LIGHTING FIXES (For the checkerboard glare) ---
        hsv_h=0.015,  # Hue shift
        hsv_s=0.7,  # Saturation shift
        hsv_v=0.4,  # Brightness shift
    )

    # ---------------------------------------------------------
    # 3. VALIDATION: Check how well it works immediately
    # ---------------------------------------------------------
    print("Training Complete. Validating now...")
    metrics = model.val()
    print(f"Final Mask mAP: {metrics.seg.map50_95}")


if __name__ == '__main__':
    main()
    # import os
    # import glob
    #
    # # Point this to your labels folder
    # label_dir = "datasets/pig-segmentation-HIT/train/labels"
    #
    # print(f"Scanning {label_dir}...")
    # for filepath in glob.glob(os.path.join(label_dir, "*.txt")):
    #     with open(filepath, "r") as f:
    #         lines = f.readlines()
    #         for i, line in enumerate(lines):
    #             parts = line.strip().split()
    #             # A segmentation mask has > 5 numbers (class + many coordinates)
    #             # A box has exactly 5 numbers (class x y w h)
    #             if len(parts) == 5:
    #                 print(f"⚠️ BAD LABEL FOUND: {filepath} (Line {i + 1})")
    #                 print("   -> This file contains a BOX, not a SEGMENT.")
    #                 print("   -> Delete this line or the whole file to fix the error.")