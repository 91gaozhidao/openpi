"""
Script to convert raw LIBERO HDF5 demo files to the LeRobot dataset v2.0 format.

This script reads the original LIBERO benchmark HDF5 demo files directly,
without requiring tensorflow_datasets or RLDS format. This is useful when
you have raw LIBERO demo files (e.g., from the LIBERO benchmark data release).

LIBERO HDF5 file structure (robosuite format):
    data/
      demo_0/
        obs/
          agentview_image: (T, H, W, 3)
          robot0_eye_in_hand_image: (T, H, W, 3)
          robot0_eef_pos: (T, 3)
          robot0_eef_quat: (T, 4)
          robot0_gripper_qpos: (T, 2)
          ...
        actions: (T, 7)
      demo_1/
        ...

Usage:
    uv run examples/libero/convert_libero_hdf5_to_lerobot.py \
        --data_dir /path/to/libero/hdf5/files \
        --repo_id your_hf_username/libero

If you want to push your dataset to the Hugging Face Hub:
    uv run examples/libero/convert_libero_hdf5_to_lerobot.py \
        --data_dir /path/to/libero/hdf5/files \
        --repo_id your_hf_username/libero \
        --push_to_hub

Note: to run this script, you need to install h5py:
    uv pip install h5py

Example for the user's data:
    uv run examples/libero/convert_libero_hdf5_to_lerobot.py \
        --data_dir /data1/shared_workspace/dataset/libero/libero_10 \
        --repo_id your_hf_username/libero

The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
"""

import math
from pathlib import Path
import re
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle representation.

    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def extract_task_name_from_filename(filename: str) -> str:
    """Extract a human-readable task description from a LIBERO HDF5 filename.

    Converts filenames like:
        KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
    to:
        turn on the stove and put the moka pot on it

    Args:
        filename: Name of the HDF5 file (without path).

    Returns:
        Human-readable task description string.
    """
    # Remove the file extension and trailing "_demo"
    name = filename.replace(".hdf5", "").replace("_demo", "")

    # Remove the scene prefix (e.g., "KITCHEN_SCENE3_", "LIVING_ROOM_SCENE1_", "STUDY_SCENE1_")
    name = re.sub(r"^[A-Z_]+SCENE\d+_", "", name)

    # Replace underscores with spaces
    return name.replace("_", " ")


def main(
    data_dir: str,
    repo_id: str = "physical-intelligence/libero",
    *,
    push_to_hub: bool = False,
    fps: int = 10,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
):
    """Convert raw LIBERO HDF5 demo files to LeRobot format.

    Args:
        data_dir: Path to directory containing LIBERO HDF5 demo files.
        repo_id: Repository ID for the output dataset.
        push_to_hub: Whether to push the dataset to the Hugging Face Hub.
        fps: Frames per second for the dataset.
        image_writer_threads: Number of threads for image writing.
        image_writer_processes: Number of processes for image writing.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all HDF5 files in the data directory
    hdf5_files = sorted(data_path.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in: {data_dir}")

    print(f"Found {len(hdf5_files)} HDF5 files in {data_dir}")

    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # Inspect first HDF5 file to get image dimensions
    with h5py.File(hdf5_files[0], "r") as f:
        demo_keys = [k for k in f["data"] if k.startswith("demo_")]
        first_demo = f["data"][demo_keys[0]]
        img_shape = first_demo["obs"]["agentview_image"].shape[1:]  # (H, W, C)
        print(f"Image shape: {img_shape}")
        print(f"Demos in first file: {len(demo_keys)}")

    # Create LeRobot dataset with features matching the existing LIBERO RLDS conversion
    # The keys here match what the training config expects after the repack transform
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": img_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    # Process each HDF5 file
    total_episodes = 0
    for hdf5_file in tqdm.tqdm(hdf5_files, desc="Processing HDF5 files"):
        # Extract task name from filename
        task_name = extract_task_name_from_filename(hdf5_file.name)
        print(f"\nProcessing: {hdf5_file.name}")
        print(f"  Task: {task_name}")

        with h5py.File(hdf5_file, "r") as f:
            # Get all demo keys and sort them numerically
            demo_keys = sorted(
                [k for k in f["data"] if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1]),
            )
            print(f"  Demos: {len(demo_keys)}")

            for demo_key in tqdm.tqdm(demo_keys, desc=f"  {hdf5_file.stem}", leave=False):
                demo = f["data"][demo_key]

                # Extract observations
                agentview_images = demo["obs"]["agentview_image"][:]
                wrist_images = demo["obs"]["robot0_eye_in_hand_image"][:]
                eef_pos = demo["obs"]["robot0_eef_pos"][:]
                eef_quat = demo["obs"]["robot0_eef_quat"][:]
                gripper_qpos = demo["obs"]["robot0_gripper_qpos"][:]
                actions = demo["actions"][:]

                num_frames = len(actions)

                for i in range(num_frames):
                    # Rotate images 180 degrees to match training preprocessing
                    # (same as in examples/libero/main.py evaluation script)
                    img = np.ascontiguousarray(agentview_images[i][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(wrist_images[i][::-1, ::-1])

                    # Build state vector: [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]
                    # This matches the 8-dim state format used in the existing conversion
                    axis_angle = quat2axisangle(eef_quat[i])
                    state = np.concatenate(
                        [
                            eef_pos[i],  # 3D position
                            axis_angle,  # 3D orientation (axis-angle)
                            gripper_qpos[i],  # 2D gripper (both finger joints)
                        ]
                    ).astype(np.float32)

                    dataset.add_frame(
                        {
                            "image": img,
                            "wrist_image": wrist_img,
                            "state": state,
                            "actions": actions[i].astype(np.float32),
                            "task": task_name,
                        }
                    )

                dataset.save_episode()
                total_episodes += 1

    print(f"\nConversion complete! Total episodes: {total_episodes}")
    print(f"Dataset saved to: {output_path}")

    dataset.consolidate()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
