import os
import numpy as np
from pathlib import Path
import gr00t
from gr00t.policy.server_client import PolicyClient
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from transformers import AutoModel, AutoProcessor
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy

# change the following paths
MODEL_PATH = "/media/lxx/Elements/project/Isaac-GR00T/checkpoints/nv-community/GR00T-N1.6-3B"

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "demo_datasets/gr1.PickNPlace")
EMBODIMENT_TAG = "gr1"

device = "cpu"
policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EmbodimentTag(EMBODIMENT_TAG),
    device=device,
    strict=True,
)
modality_config = policy.get_modality_config()

# processor = AutoProcessor.from_pretrained(Path(MODEL_PATH))
# modality_config = processor.get_modality_configs()[EMBODIMENT_TAG]

# Create the dataset
dataset = LeRobotEpisodeLoader(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="torchcodec",
    video_backend_kwargs=None,
)

import numpy as np

episode_data = dataset[0]
step_data = extract_step_data(
    episode_data, step_index=0, modality_configs=modality_config, embodiment_tag=EmbodimentTag(EMBODIMENT_TAG), allow_padding=False
)

observation = {
    "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},  # stach images and add batch dimension
    "state": {k: step_data.states[k][None] for k in step_data.states},  # add batch dimension
    "action": {k: step_data.actions[k][None] for k in step_data.actions},  # add batch dimension
    "language": {
        modality_config["language"].modality_keys[0]: [[step_data.text]],  # add time and batch dimension
    }
}



# Connect to the policy server
policy = PolicyClient(
    host="localhost",  # or IP address of your GPU server
    port=5555,
    timeout_ms=15000,  # 15 second timeout for inference
    strict=False,      # leave the validation to the server
)

# Verify connection
if not policy.ping():
    raise RuntimeError("Cannot connect to policy server!")

predicted_action, info = policy.get_action(observation)

for key, value in predicted_action.items():
    print(key, value.shape)


