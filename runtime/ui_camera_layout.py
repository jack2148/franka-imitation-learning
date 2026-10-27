"""Display-only Geomagic Square camera selection; never dataset overrides."""
from copy import deepcopy

FRONT_PREVIEW_CAMERAS = ("operator", "front_operator", "front")
# The XML's "front" is top-down. Reframe only the preview process's copy from
# +X toward the workspace; image right is +Y and image up retains +Z.
# Look-at: (0.50, 0.03, 0.50). Existing XML/dataset cameras remain untouched.
FRONT_PREVIEW_OVERRIDE = {
    "pos": (1.70, 0.03, 1.15),
    "quat": (0.6075119646024991, 0.36184141949866916,
             0.36184141949866916, 0.6075119646024991),
    "fovy": 48.0,
}


def geomagic_square_front_panel(leader, source, task):
    return (leader == "geomagic" and source == "physical_hdapi"
            and task in ("mimicgen_square", "fr3-mimicgen-square"))


def front_preview_overrides(task_overrides):
    overrides = deepcopy(task_overrides or {})
    overrides["front"] = deepcopy(FRONT_PREVIEW_OVERRIDE)
    return overrides


def front_camera_state():
    return {
        "primary_view": "operator",
        "bottom_right_view": "front",
        "message": "Operator View + Top-Down Alignment · EE Point + Front View",
        "operator_frame_url": "/api/work/frame?camera=operator",
        "front_operator_frame_url": "/api/work/frame?camera=front_operator",
        "front_frame_url": "/api/work/frame?camera=front",
    }
