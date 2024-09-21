import torch as th

from stable_baselines3 import PPO

import onnx
import onnx_tf.backend
import tensorflow as tf


class OnnxablePolicy(th.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        batch_size, _, _ = observation.shape

        observation_flat = observation.view(batch_size, -1)

        action_hidden, value_hidden = self.extractor(observation_flat)
        return self.action_net(action_hidden), self.value_net(value_hidden)


model = PPO.load("models/PPO_21/3980000.zip", device="cpu")
onnxable_model = OnnxablePolicy(
    model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
)

observation_size = model.observation_space.shape
dummy_input = th.randn(1, *observation_size)
th.onnx.export(
    onnxable_model,
    dummy_input,
    "model.onnx",
    opset_version=9,
    input_names=["input"],
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
