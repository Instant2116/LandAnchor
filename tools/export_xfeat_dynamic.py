import torch
from modules.xfeat import XFeat

# 1. Initialize
xfeat_wrapper = XFeat()
model = xfeat_wrapper.net
model.eval()

# 2. Define Resolution Constraints
# min=160 (20*8), max=1080 (135*8)
_h = torch.export.Dim("_h", min=20, max=135)
_w = torch.export.Dim("_w", min=20, max=240)

# We define the batch as a plain integer (1) to fix it
# and height/width as dynamic multiples of 8.
dynamic_shapes = {
    "x": {
        0: 1,           # Fixed Batch size 1
        2: 8 * _h,      # Dynamic Height (multiple of 8)
        3: 8 * _w       # Dynamic Width (multiple of 8)
    }
}

# 3. Export
dummy_input = torch.randn(1, 3, 480, 640)
output_onnx = "xfeat_smart_dynamic_v2.onnx"

print("Starting Smart Export with fixed batch and dynamic resolution...")

torch.onnx.export(
    model,
    (dummy_input,),
    output_onnx,
    dynamo=True,
    dynamic_shapes=dynamic_shapes,
    input_names=['input'],
    output_names=['feats', 'keypoints', 'scores'],
    opset_version=18
)

print(f"Success! Dynamic model exported to {output_onnx}")