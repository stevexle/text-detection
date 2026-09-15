"""
End-to-end Pipeline Modules for Vietnamese ID Card (CCCD) Processing.
"""

from src.pipeline.cccd_pipeline import CCCDDetectionPipeline
from src.pipeline.cccd_pipeline_onnx import CCCDDetectionPipelineONNX
from src.pipeline.cccd_pipeline_trt import CCCDDetectionPipelineTRT
from src.pipeline.spatial_matcher import match_text_to_fields

__all__ = [
    "CCCDDetectionPipeline",
    "CCCDDetectionPipelineONNX",
    "CCCDDetectionPipelineTRT",
    "match_text_to_fields",
]
