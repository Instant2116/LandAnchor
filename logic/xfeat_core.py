import numpy as np
import onnxruntime as ort

class XFeatCore:
    def __init__(self, model_path, gem_p=3):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.gem_p = gem_p

    def set_gem_p(self, new_p):
        """Allows dynamic updating of the pooling power from the UI."""
        self.gem_p = new_p

    def process(self, img_320x320):
        # 1. Inference
        tensor = (
            np.expand_dims(np.transpose(img_320x320, (2, 0, 1)), axis=0).astype(
                np.float32
            )
            / 255.0
        )
        outputs = self.session.run(None, {"input": tensor})

        # 2. Extract and Normalize
        desc = outputs[0][0].reshape(64, -1).T
        desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-6

        # 3. GeM Pooling
        # Calculate the mean of the powered descriptors dynamically using self.gem_p
        mean_powered = np.mean(np.power(desc, self.gem_p), axis=0)

        # Extract the sign and apply the fractional root only to the absolute values
        gem = np.sign(mean_powered) * np.power(np.abs(mean_powered), 1.0 / self.gem_p)
        gem /= np.linalg.norm(gem) + 1e-6
        gem /= np.linalg.norm(gem) + 1e-6

        return {
            "kpts": outputs[1][0],
            "desc": desc,
            "global": gem,
            "scores": outputs[2][0],
        }