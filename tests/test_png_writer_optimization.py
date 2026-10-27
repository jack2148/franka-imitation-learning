"""Core PNG/JPEG byte fidelity; extension recorder tests are outside v1."""
from pathlib import Path
import io,tempfile,unittest
import numpy as np
from PIL import Image
from runtime.recording import _save_image
class PngWriterTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
 def tearDown(self):
  self.temp.cleanup()
 def test_png_zero_is_exact_rgb_and_jpeg_settings_unchanged(self):
     rgb = np.random.default_rng(77).integers(0, 256, (480, 640, 3), dtype=np.uint8)
     path = self.root / 'frame.png'
     _save_image(rgb, path, 'PNG')
     with Image.open(path) as im:
         self.assertTrue(np.array_equal(np.asarray(im), rgb))
         self.assertEqual(im.mode, 'RGB')
     expected = io.BytesIO()
     Image.fromarray(rgb).save(expected, format='PNG', compress_level=0)
     self.assertEqual(path.read_bytes(), expected.getvalue())
     jpeg = self.root / 'preview.jpg'
     _save_image(rgb, jpeg, 'JPEG')
     expected = io.BytesIO()
     Image.fromarray(rgb).save(expected, format='JPEG', quality=92)
     self.assertEqual(jpeg.read_bytes(), expected.getvalue())
