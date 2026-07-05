from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import tensorrt as trt


def _trt_dtype_to_torch_dtype(dtype: trt.DataType) -> torch.dtype:
    """Map TensorRT dtype to torch dtype.

    Written defensively for TensorRT 10/11 API differences.
    """
    dtype_str = str(dtype).lower()

    if "float" in dtype_str and "16" not in dtype_str:
        return torch.float32
    if "half" in dtype_str or "float16" in dtype_str or "fp16" in dtype_str:
        return torch.float16
    if "bfloat16" in dtype_str or "bf16" in dtype_str:
        return torch.bfloat16
    if "int8" in dtype_str:
        return torch.int8
    if "int32" in dtype_str:
        return torch.int32
    if "int64" in dtype_str:
        return torch.int64
    if "bool" in dtype_str:
        return torch.bool

    # Fallback for common TRT aliases.
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int8:
        return torch.int8
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.bool:
        return torch.bool

    if hasattr(trt, "bfloat16") and dtype == trt.bfloat16:
        return torch.bfloat16

    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


class Engine:
    """Small TensorRT engine wrapper using execute_async_v3.

    Input tensors are passed by keyword. Output tensors are returned in a dict.
    """

    def __init__(self, engine_path: str | Path):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.logger = trt.Logger(trt.Logger.INFO)

        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context: {self.engine_path}")

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.name_to_dtype: Dict[str, torch.dtype] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)

            self.name_to_dtype[name] = _trt_dtype_to_torch_dtype(dtype)

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
            else:
                raise RuntimeError(f"Unknown TensorRT tensor mode for {name}: {mode}")

        print(f"[TRT] Loaded engine: {self.engine_path}")
        print(f"[TRT]   inputs:  {self.input_names}")
        print(f"[TRT]   outputs: {self.output_names}")

    def _prepare_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        expected_dtype = self.name_to_dtype[name]

        if not tensor.is_cuda:
            tensor = tensor.cuda()

        # Do not blindly cast integer/bool tensors to floating point.
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(dtype=expected_dtype)

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor

    @torch.inference_mode()
    def __call__(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        missing = [name for name in self.input_names if name not in kwargs]
        if missing:
            raise KeyError(
                f"Missing TensorRT inputs for {self.engine_path.name}: {missing}. "
                f"Provided: {list(kwargs.keys())}"
            )

        inputs: Dict[str, torch.Tensor] = {}

        for name in self.input_names:
            tensor = self._prepare_input(name, kwargs[name])
            inputs[name] = tensor

            # Set dynamic input shape.
            ok = self.context.set_input_shape(name, tuple(tensor.shape))
            if not ok:
                raise RuntimeError(
                    f"TensorRT rejected input shape for {name}: "
                    f"got {tuple(tensor.shape)} in engine {self.engine_path}"
                )

        # Some TRT versions expose infer_shapes(). Use it if available.
        if hasattr(self.context, "infer_shapes"):
            self.context.infer_shapes()

        outputs: Dict[str, torch.Tensor] = {}

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Output shape for {name} is still dynamic after setting inputs: {shape}"
                )

            dtype = self.name_to_dtype[name]
            outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)

        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        stream = torch.cuda.current_stream()
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path}")

        return outputs
