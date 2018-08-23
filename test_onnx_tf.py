import torch
from torch import nn
import tensorflow as tf
import numpy as np
import onnx
import onnx.utils
from onnx_tf.backend import prepare
from typing import List, Tuple
from pathlib import Path
from pytest import approx

def helper_test_onnx_tf(model, input_shape: List[int], tmpdir: str):
    """Ensure the model outputs do not change under onnx-tf transformation."""
    # run the model and cache the outputs
    test_spec: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(5):
        rnd_input = np.random.rand(*input_shape).astype(np.float32)
        spec_output = model(torch.from_numpy(rnd_input))
        test_spec.append((rnd_input, spec_output))
    # convert model to tensorflow graph
    onnx_model_path = str(Path(tmpdir) / "model.onnx")
    torch.onnx.export(model, torch.from_numpy(test_spec[0][0]), onnx_model_path)
    original_onnx_model = onnx.load(onnx_model_path)
    onnx_model = onnx.optimizer.optimize(original_onnx_model, passes=[
        'eliminate_identity',
        'eliminate_nop_transpose',
        'fuse_consecutive_transposes',
        'fuse_consecutive_squeezes',
        'fuse_add_bias_into_conv',
    ])
    onnx_model = onnx.utils.polish_model(onnx_model)
    tf.reset_default_graph()
    tf_rep = prepare(onnx_model)
    # test tf graph
    for (rnd_input, spec_output) in test_spec:
        tf_output = tf_rep.run({"0": rnd_input})
        assert tf_output._0 == approx(spec_output.data.numpy(), abs=1e-4, rel=1e-4)

def test_basic_add(tmpdir):
    class Model(nn.Module):
        def forward(self, x):
            return x + 1
    helper_test_onnx_tf(Model(), input_shape=[3, 5, 7], tmpdir=tmpdir)

def test_basic_mul(tmpdir):
    class Model(nn.Module):
        def forward(self, x):
            return x * 3
    helper_test_onnx_tf(Model(), input_shape=[8, 1], tmpdir=tmpdir)

def test_conv(tmpdir):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.m = nn.Conv2d(16, 22, 3, stride=2)
        def forward(self, x):
            return self.m(x)
    helper_test_onnx_tf(Model(), input_shape=[7, 16, 33, 33], tmpdir=tmpdir)

def test_gru(tmpdir):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.gru = nn.GRU(16, 128, num_layers=1, batch_first=False)
        def forward(self, x):
            output, _ = self.gru(x)
            return output
    helper_test_onnx_tf(Model(), input_shape=[7, 3, 16], tmpdir=tmpdir)

def test_bidi_gru(tmpdir):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.gru = nn.GRU(16, 128, num_layers=1, batch_first=False, bidirectional=True)
        def forward(self, x):
            output, _ = self.gru(x)
            return output
    helper_test_onnx_tf(Model(), input_shape=[7, 3, 16], tmpdir=tmpdir)

def test_stack_gru(tmpdir):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.gru = nn.GRU(16, 128, num_layers=3, batch_first=False)
        def forward(self, x):
            output, _ = self.gru(x)
            return output
    helper_test_onnx_tf(Model(), input_shape=[7, 3, 16], tmpdir=tmpdir)

def test_stack_bidi_gru(tmpdir):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.gru = nn.GRU(16, 128, num_layers=3, batch_first=False, bidirectional=True)
        def forward(self, x):
            output, _ = self.gru(x)
            return output
    helper_test_onnx_tf(Model(), input_shape=[7, 3, 16], tmpdir=tmpdir)

def test_lstm(tmpdir):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.lstm = nn.LSTM(16, 128, num_layers=1, batch_first=False)
        def forward(self, x):
            output, _ = self.lstm(x)
            return output
    helper_test_onnx_tf(Model(), input_shape=[7, 3, 16], tmpdir=tmpdir)

def test_bidi_lstm(tmpdir):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.lstm = nn.LSTM(16, 128, num_layers=1, batch_first=False, bidirectional=True)
        def forward(self, x):
            output, _ = self.lstm(x)
            return output
    helper_test_onnx_tf(Model(), input_shape=[7, 3, 16], tmpdir=tmpdir)

def test_stack_lstm(tmpdir):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.lstm = nn.LSTM(16, 128, num_layers=3, batch_first=False)
        def forward(self, x):
            output, _ = self.lstm(x)
            return output
    helper_test_onnx_tf(Model(), input_shape=[7, 3, 16], tmpdir=tmpdir)

def test_stack_bidi_lstm(tmpdir):
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.lstm = nn.LSTM(16, 128, num_layers=3, batch_first=False, bidirectional=True)
        def forward(self, x):
            output, _ = self.lstm(x)
            return output
    helper_test_onnx_tf(Model(), input_shape=[7, 3, 16], tmpdir=tmpdir)
