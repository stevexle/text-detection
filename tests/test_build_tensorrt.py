"""
Unit tests for NVIDIA TensorRT Engine Builder and Pipeline.
Validates shape profiles, command construction, dry-run CLI execution, and runner fallbacks.
"""

from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest

from tools.build_tensorrt import SHAPE_PROFILES, construct_trtexec_cmd, find_trtexec


class TestTensorRTBuilder:
    def test_shape_profiles_definitions(self):
        """Verify dynamic shape profiles for DBNet, YOLO-seg, and YOLO-cls."""
        assert "dbnet" in SHAPE_PROFILES
        assert "yolo_seg" in SHAPE_PROFILES
        assert "yolo_cls" in SHAPE_PROFILES

        db_prof = SHAPE_PROFILES["dbnet"]
        assert db_prof["input_name"] == "input"
        assert db_prof["min"] == "1x3x480x480"
        assert db_prof["opt"] == "1x3x960x704"
        assert db_prof["max"] == "8x3x960x960"

        seg_prof = SHAPE_PROFILES["yolo_seg"]
        assert seg_prof["input_name"] == "images"
        assert seg_prof["min"] == "1x3x640x640"
        assert seg_prof["opt"] == "1x3x640x640"

    def test_construct_trtexec_cmd_fp16(self):
        """Verify trtexec command construction with FP16 precision."""
        cmd = construct_trtexec_cmd(
            trtexec_bin="/usr/bin/trtexec",
            onnx_path="weights/onnx/dbnet.onnx",
            engine_path="weights/tensorrt/dbnet.engine",
            input_name="input",
            min_shape="1x3x480x480",
            opt_shape="1x3x960x704",
            max_shape="8x3x960x960",
            fp16=True,
            workspace_mb=2048,
        )

        assert "/usr/bin/trtexec" == cmd[0]
        assert "--onnx=weights/onnx/dbnet.onnx" in cmd
        assert "--saveEngine=weights/tensorrt/dbnet.engine" in cmd
        assert "--minShapes=input:1x3x480x480" in cmd
        assert "--optShapes=input:1x3x960x704" in cmd
        assert "--maxShapes=input:8x3x960x960" in cmd
        assert "--memPoolSize=workspace:2048M" in cmd
        assert "--fp16" in cmd
        assert "--int8" not in cmd

    def test_construct_trtexec_cmd_int8(self):
        """Verify trtexec command construction with INT8 precision."""
        cmd = construct_trtexec_cmd(
            trtexec_bin="trtexec",
            onnx_path="weights/onnx/yolo26_seg.onnx",
            engine_path="weights/tensorrt/yolo26_seg.engine",
            input_name="images",
            min_shape="1x3x640x640",
            opt_shape="1x3x640x640",
            max_shape="8x3x640x640",
            fp16=False,
            int8=True,
            workspace_mb=4096,
        )

        assert "--int8" in cmd
        assert "--fp16" not in cmd
        assert "--memPoolSize=workspace:4096M" in cmd

    def test_find_trtexec_mocked(self):
        """Verify find_trtexec locates binary on PATH when present."""
        with patch("shutil.which", return_value="/custom/bin/trtexec"):
            found = find_trtexec()
            assert found == "/custom/bin/trtexec"

    def test_find_trtexec_not_found(self):
        """Verify find_trtexec returns None when binary is nowhere on system."""
        with patch("shutil.which", return_value=None):
            with patch("os.access", return_value=False):
                found = find_trtexec()
                assert found is None

    def test_cli_dry_run_execution(self):
        """Test build_tensorrt.py dry-run execution via subprocess."""
        cmd = [
            sys.executable,
            "tools/build_tensorrt.py",
            "--model",
            "dbnet",
            "--dry-run",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "[Dry Run]" in res.stdout
        assert "TensorRT" in res.stdout

    def test_cli_dry_run_with_trtexec_flag(self):
        """Test build_tensorrt.py dry-run execution with explicit trtexec flag."""
        cmd = [
            sys.executable,
            "tools/build_tensorrt.py",
            "--model",
            "dbnet",
            "--trtexec-path",
            "/usr/bin/trtexec",
            "--dry-run",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "[Dry Run]" in res.stdout
        assert "trtexec" in res.stdout


class TestCCCDPipelineTRTImports:
    def test_pipeline_class_import(self):
        """Verify CCCDDetectionPipelineTRT can be imported cleanly."""
        from src.pipeline.cccd_pipeline_trt import CCCDDetectionPipelineTRT
        assert CCCDDetectionPipelineTRT is not None
