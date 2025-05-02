
import torch as th

from stable_baselines3 import PPO

import onnx
import onnx_tf.backend
import tensorflow as tf


class OnnxablePolicy(th.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, distance, local, obstacle_ahead):
        obs_dict = {
            "distance": distance,
            "local": local,
            "obstacle_ahead": obstacle_ahead
        }

        features = self.policy.extract_features(obs_dict)

        action_latent, value_latent = self.policy.mlp_extractor(features)
        action_logits = self.policy.action_net(action_latent)
        value = self.policy.value_net(value_latent)

        return action_logits, value

model = PPO.load("models/PPO_25/4560000.zip", device="cpu")
onnxable_model = OnnxablePolicy(model.policy)
dummy_input = {
    key: th.randn(1, *space.shape)
    for key, space in model.observation_space.spaces.items()
}

th.onnx.export(
    onnxable_model,
    args=(dummy_input["distance"], dummy_input["local"], dummy_input["obstacle_ahead"]), 
    f="model.onnx",
    input_names=["distance","local", "obstacle_ahead"],
    output_names=["action_logits", "value"],
    opset_version=11,
    dynamic_axes={
        "distance": {0: "batch"},
        "local": {0: "batch"},
        "obstacle_ahead": {0: "batch"},
        "action_logits": {0: "batch"},
        "value": {0: "batch"}
    }
)

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph("model")

print('Converting TF to TFLite...')
converter = tf.lite.TFLiteConverter.from_saved_model("model")
tflite_model = converter.convert()
with open("model.tflite", 'wb') as f:
    f.write(tflite_model)
