import torch
from modules.xfeat import XFeat

# Initialize
xfeat_wrapper = XFeat()
model = xfeat_wrapper.net
model.eval()

# Static Input 320x320
dummy_input = torch.randn(1, 3, 320, 320)
output_onnx = "xfeat_static_320.onnx"

print("Exporting static 320x320 model...")

torch.onnx.export(
    model,
    dummy_input,
    output_onnx,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['feats', 'keypoints', 'scores'],
    dynamic_axes=None  # No dynamic axes for static
)

print(f"Static model successfully exported to {output_onnx}")