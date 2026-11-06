"""
segmentation_timed.py -- subclasses segmentation.SAM2Segmenter (that file
is left completely untouched) to time it, specifically separating the
interactive click step from everything automated around it.

Model loading (__init__), the interactive click/name prompt
(_get_user_clicks_with_names -- matplotlib click loop + input() calls, pure
human wall-clock time), and segment_scene()'s total wall-clock time are all
recorded into self.timing and returned as results['timing'] from
segment_scene(). "automated_processing_sec" = total - clicks, i.e. the SAM2
inference/propagation/merge/save work that isn't waiting on a person.

Only these two methods are overridden -- both call straight through to the
real implementation via super(), so segmentation behavior itself is
byte-for-byte identical to segmentation.py; nothing about how masks are
produced changes here, only what gets timed around it.
"""
import time

from segmentation import SAM2Segmenter


class SAM2SegmenterTimed(SAM2Segmenter):

    def __init__(self, config: dict):
        _t0 = time.perf_counter()
        super().__init__(config)
        self.timing = {"model_load_sec": round(time.perf_counter() - _t0, 3)}

    def _get_user_clicks_with_names(self, image):
        _t0 = time.perf_counter()
        result = super()._get_user_clicks_with_names(image)
        self.timing["clicks_sec"] = round(time.perf_counter() - _t0, 3)
        return result

    def segment_scene(self, image_dir: str, output_dir: str):
        _t0 = time.perf_counter()
        result = super().segment_scene(image_dir, output_dir)
        total = time.perf_counter() - _t0
        self.timing["segment_scene_total_sec"] = round(total, 3)
        self.timing["automated_processing_sec"] = round(
            total - self.timing.get("clicks_sec", 0.0), 3)
        result["timing"] = dict(self.timing)
        return result
