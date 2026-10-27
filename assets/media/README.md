# Square demonstration media

These previews show recorded simulator views from physical Geomagic Touch input controlling a MuJoCo-simulated Franka FR3. The device and user are not visible. They document training-data collection, not policy inference or a TEST50 success-rate measurement.

## Source and identity

The locally verified collection metadata identifies episode `001_1539` (collection order 0, processed episode 0), recorded on 2026-10-08, with 928 samples at 20 Hz. It records `leader_device=geomagic_touch`, `input_source=physical_hdapi`, and `synthetic_input=false`. The episode belongs to `square_geomagic_xy50_v1`, converted as `square_geomagic_xy50_v1_convert`: 50 demonstrations / 48,272 frames. See the public [dataset metadata identity](../../results/square_test50/DATASET_IDENTITY.json).

The source review video, `001_1539.mp4`, was already encoded from the episode's saved operator and wrist PNGs. It is 46.4 seconds at 20 fps, 1280×544, with operator view on the left, wrist view on the right, and an existing episode/frame/simulation-time header. Source video SHA256:

```text
b08a236b36f6256f04aeb1a1d504d4127c11a2d0626a2716e7eee2188373a286
```

This attribution was checked against the archived media handoff on 2026-10-27. The underlying collection metadata and conversion record remain external; their SHA256 values are recorded below for identification, not as substitutes for the unavailable full payloads:

```text
Episode metadata.json:
d8ed0d3f531671411a14c37239dc3dbcfba40e00875905901d2b448089b4aa05
Processed meta/mimic_conversion.json:
461dba987c1738a7022b81df4db80d2f1dc4f26588fd7388f373606467d2687e
```

## Published previews

Both files are byte-identical copies of the verified handoff previews. They show source playback interval **[30, 42) seconds**, preserving 1× simulation-sample timing. They are not a measurement of human wall-clock timing or system latency.

| File | Format | Size |
| --- | --- | ---: |
| [GIF preview](geomagic_touch_square_collection.gif) | 720×306, 10 fps, 12 s, looping | 2,649,333 bytes |
| [MP4 preview](geomagic_touch_square_collection.mp4) | H.264/yuv420p, 1280×544, 20 fps, 12 s | 627,769 bytes |

```text
39ac54a2b50919542f1ef10a87fba9d341052f75672ae736d655a3c93310c28f  geomagic_touch_square_collection.gif
c6f36454557ffeb68ac9d2d00d9593daeb431e082476f1df82b7561b378a9723  geomagic_touch_square_collection.mp4
```

The handoff preparation trimmed the existing video, encoded the MP4 at CRF20, and proportionally resized/downsampled the GIF with a palette and dithering. Neither preview crops the scene or adds overlays. The MP4 contains no audio; container metadata and chapters were removed. No new simulation rendering or rollout was used to produce these previews.

The excerpt does not show the whole demonstration or establish task success. It is unrelated to the later 50,822-frame MP4 collection and does not demonstrate physical Franka deployment or physical haptic acceptance. See [reproducibility limits](../../REPRODUCIBILITY.md).
