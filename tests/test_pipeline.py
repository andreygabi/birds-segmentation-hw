from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from predict import predict
from src.data import find_pairs, image_files, letterbox, read_image, read_mask
from src.metrics import binary_mask, image_iou
from src.model import UNet


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_iou_edge_cases(self):
        empty = np.zeros((3, 4), bool)
        full = np.ones((3, 4), bool)
        self.assertEqual(image_iou(empty, empty), 1)
        self.assertEqual(image_iou(empty, full), 0)
        self.assertEqual(image_iou(full, empty), 0)
        self.assertEqual(image_iou([[1, 0], [0, 1]], [[1, 1], [0, 0]]), 1 / 3)
        with self.assertRaises(ValueError):
            image_iou(empty, empty.T)

    def test_readers_and_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images/bird").mkdir(parents=True)
            (root / "gt/bird").mkdir(parents=True)
            Image.fromarray(np.zeros((9, 13), np.uint8)).save(root / "images/bird/a.jpeg")
            target = np.zeros((9, 13, 3), np.uint8)
            target[2:5, 3:8, 0] = 255
            target[..., 1] = 255
            target[0, 0, 0] = 204
            Image.fromarray(target).save(root / "gt/bird/a.png")
            pair, = find_pairs(root / "images", root / "gt")
            self.assertEqual(pair.key, "bird/a")
            self.assertEqual(read_image(pair.image).mode, "RGB")
            self.assertEqual(read_mask(pair.mask).sum(), 15)
            Image.new("RGB", (13, 9)).save(root / "images/bird/a.png")
            with self.assertRaises(ValueError):
                image_files(root / "images")

    def test_restore_and_prediction_without_masks(self):
        image = Image.new("RGB", (47, 19))
        _, _, meta = letterbox(image, 64)
        mask = binary_mask(torch.full((1, 64, 64), 8.0), meta, .5)
        self.assertEqual(mask.shape, (19, 47))
        self.assertTrue((mask == 1).all())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_root = root / "only_images/bird"
            images_root.mkdir(parents=True)
            Image.new("L", (47, 19), 120).save(images_root / "gray.jpg")
            model = UNet(pretrained=False).eval()
            checkpoint = root / "random.pt"
            torch.save({"model_state": model.state_dict(), "config": {"image_size": 64}, "threshold": .5}, checkpoint)
            predict(checkpoint, root / "only_images", root / "out", "cpu", batch_size=1, threads=2)
            with Image.open(root / "out/bird/gray.png") as result:
                self.assertEqual(result.mode, "L")
                self.assertEqual(result.size, (47, 19))
                self.assertTrue(np.isin(np.asarray(result), [0, 255]).all())
            self.assertFalse((root / "gt").exists())

    def test_unet_odd_input_dimensions(self):
        model = UNet(pretrained=False).eval()
        with torch.inference_mode():
            result = model(torch.randn(1, 3, 65, 97))
        self.assertEqual(tuple(result.shape), (1, 1, 65, 97))


if __name__ == "__main__":
    unittest.main()
