import numpy as np
import tensorflow as tf


# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="snake1_tf_agents.tflite")
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Test model on random input data.
# input_shape = input_details[2]['shape']
# input_data_type = input_details[2]['dtype']

# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# print(output_details[0])
# print(input_data)
# interpreter.set_tensor(input_details[2]['index'], input_data)
# interpreter.invoke()


# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)

# model = tf.saved_model.load('model1') 
# print(type(model))
# input_shape = (1, 3)
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# input_data = np.array([[100,100,100]])
# print(input_data)
# print(list(model.signatures.keys())) 
# infer = model.signatures["action"]
# print(infer.structured_outputs)
# Prepare the required input arguments
input_data = np.array([[100, 100, 100]])
observation = tf.constant(input_data, dtype=tf.float32)
step_type = tf.constant([0], dtype=tf.int32)  # Wrap the step_type in a list to match the expected shape
discount = tf.constant([0.0], dtype=tf.float32)  # Wrap the discount in a list to match the expected shape
reward = tf.constant([0.0], dtype=tf.float32)  # Wrap the reward in a list to match the expected shape


# feed_dict = {'0/observation' : observation, '0/step_type': step_type, '0/discount': discount, '0/reward' : reward}
# output = infer(**feed_dict) 
# print(output)
import tf_agents
test= tf_agents.trajectories.time_step.TimeStep(
    step_type, reward, discount, observation)

saved_policy = tf.compat.v2.saved_model.load('model1')
policy_state = saved_policy.get_initial_state(batch_size=0)

policy_step = saved_policy.action(test, policy_state)
policy_state = policy_step.state
time_step =  policy_step.action
print(time_step)
value = time_step.numpy()[0]
print(value)
# predictions = model(input_data)
# print(predictions)