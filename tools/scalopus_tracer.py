import scalopus
import scalopus.tracing as tracing
import time
import json
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Dict, Any, Optional


class ScalopusWrapper:
    """Wrapper for Scalopus tracing functionality."""

    def __init__(self, process_name: str = "hi"):
        """Initialize Scalopus tracing components."""
        self.exposer = scalopus.common.DefaultExposer(process_name=process_name)
        scalopus.general.setThreadName("main")

        factory = scalopus.transport.TransportUnixFactory()
        poller = scalopus.general.EndpointManagerPoll(factory)

        # Initialize providers
        native_provider = scalopus.tracing.native.NativeTraceProvider(poller)
        general_provider = scalopus.general.GeneralProvider(poller)

        # Configure endpoint factories
        self._configure_endpoints(poller, native_provider, general_provider)
        self.poller = poller

        self.poller.startPolling(1.0)

        self.native_source = native_provider.makeSource()
        self.general_source = general_provider.makeSource()

    def _configure_endpoints(self, poller, native_provider, general_provider):
        """Helper method to configure endpoint factories."""
        endpoints = [
            (scalopus.tracing.EndpointNativeTraceSender.name, native_provider.factory),
            (
                scalopus.tracing.EndpointTraceMapping.name,
                scalopus.tracing.EndpointTraceMapping.factory,
            ),
            (
                scalopus.general.EndpointProcessInfo.name,
                scalopus.general.EndpointProcessInfo.factory,
            ),
        ]

        for name, factory in endpoints:
            poller.addEndpointFactory(name, factory)

    def start(self):
        """Start tracing intervals."""
        self.native_source.startInterval()
        self.general_source.startInterval()

    def end(self):
        """Stop tracing and save results to file."""
        time.sleep(1)  # Allow time for final events

        self.native_source.stopInterval()
        self.general_source.stopInterval()
        self.poller.stopPolling()

        try:
            data = self.native_source.finishInterval()
            data.extend(self.general_source.finishInterval())

            self._save_trace_data(data)
        except RuntimeError:
            pass

    def _save_trace_data(self, data):
        """Save trace data to JSON file with proper formatting."""
        with open("./trace.json", "w") as f:
            f.write("[\n")
            f.write(",\n".join(json.dumps(entry) for entry in data))
            f.write("\n]")


class _AtenOpTracer(TorchDispatchMode):
    """Handles tracing of PyTorch ATEN operations."""

    def __init__(self, model_tracer: "ModelTracer"):
        super().__init__()
        self._model_tracer = model_tracer

    def __torch_dispatch__(self, func, types, args={}, kwargs=None):
        torch.cuda.synchronize()
        module_type = str(func)
        print(f"[ATEN OP] module_type = {module_type}")
        ctx = self._model_tracer._create_or_get_context(module_type)
        ctx.enter()
        result = func(*args, **(kwargs or {}))
        torch.cuda.synchronize()
        ctx.exit()
        return result


class ModelTracer:
    """
    Handles tracing for a single model instance.

    Example:
        >> tracer = ModelTracer().init_scalopus("my_model")
        >> tracer.register_hooks(model)
        >> dist.all_reduce = tracer.trace_comm_ops(dist.all_reduce)
        >> tracer.trace_aten_ops()
        >> tracer.start_tracing()
        >> # ...运行模型...
        >> tracer.stop_tracing()
    """

    def __init__(self):
        self.trace_contexts: Dict[str, tracing.TraceContext] = {}
        self.scalopus_wrapper: Optional[ScalopusWrapper] = None
        self.aten_op_tracer: Optional[_AtenOpTracer] = None

    def init_scalopus(self, process_name: str = "hi"):
        """Initialize Scalopus tracing system."""
        self.scalopus_wrapper = ScalopusWrapper(process_name)
        return self

    def register_hooks(self, model: torch.nn.Module):
        """Register tracing hooks for all modules in model."""
        for _, module in model.named_modules():
            module.register_forward_pre_hook(self._make_forward_pre_hook())
            module.register_forward_hook(self._make_forward_hook())
            module.register_full_backward_pre_hook(self._make_backward_pre_hook())
            module.register_full_backward_hook(self._make_backward_hook())

    def trace_comm_ops(self, comm_op):
        """Trace communication operations."""
        return self._comm_op_wrapper(comm_op)

    def trace_aten_ops(self):
        self.aten_op_tracer = _AtenOpTracer(self)

    def start_tracing(self):
        """Start recording traces."""
        if self.scalopus_wrapper:
            self.scalopus_wrapper.start()
            if self.aten_op_tracer is not None:
                self.aten_op_tracer.__enter__()

    def stop_tracing(self):
        """Stop recording and save traces."""
        if self.scalopus_wrapper:
            if self.aten_op_tracer is not None:
                self.aten_op_tracer.__exit__(None, None, None)
            self.scalopus_wrapper.end()

    def _comm_op_wrapper(self, comm_op):
        """Wrap communication operation in a trace context."""

        def wrapper(*args, **kwargs):
            print(
                f"[Communication Op] {comm_op.__name__} called, "
                f"group: {dist.get_process_group_ranks(kwargs.get('group', 'WORLD'))}, "
                # f"args: {args}, kwargs: {kwargs}"
            )
            torch.cuda.synchronize()
            module_type = comm_op.__name__
            ctx = self._create_or_get_context(module_type)
            ctx.enter()
            result = comm_op(*args, **kwargs)
            torch.cuda.synchronize()
            ctx.exit()
            return result

        return wrapper

    def _get_module_type(self, module: torch.nn.Module) -> str:
        """Extract clean module type name from module object."""
        return str(type(module)).split("'")[1]

    def _create_or_get_context(self, key: str) -> tracing.TraceContext:
        """Get or create a trace context for the given module type."""
        if key not in self.trace_contexts:
            self.trace_contexts[key] = tracing.TraceContext(key)
        return self.trace_contexts[key]

    def _log_event(self, event: str, module_type: str):
        """Helper function to log module events."""
        print(f"[INFO] {event}: module_type: {module_type}")

    def _make_forward_pre_hook(self):
        def hook(module: torch.nn.Module, data_input: Any):
            torch.cuda.synchronize()
            module_type = self._get_module_type(module)
            ctx = self._create_or_get_context(module_type)
            ctx.enter()
            self._log_event("forward_hook_pre", module_type)

        return hook

    def _make_forward_hook(self):
        def hook(module: torch.nn.Module, data_input: Any, data_output: Any):
            """Forward pass hook for tracing."""
            torch.cuda.synchronize()
            module_type = self._get_module_type(module)
            ctx = self.trace_contexts[module_type]
            ctx.exit()
            self._log_event("forward_hook_pos", module_type)

        return hook

    def _make_backward_pre_hook(self):
        def hook(module: torch.nn.Module, data_input: Any):
            """Backward pass pre-hook for tracing."""
            torch.cuda.synchronize()
            module_type = self._get_module_type(module)
            module_type = f"{module_type}.grad"
            ctx = self._create_or_get_context(module_type)
            ctx.enter()
            self._log_event("backward_hook_pre", module_type)

        return hook

    def _make_backward_hook(self):
        def hook(module: torch.nn.Module, data_input: Any, data_output: Any):
            """Backward pass hook for tracing."""
            torch.cuda.synchronize()
            module_type = self._get_module_type(module)
            module_type = f"{module_type}.grad"
            ctx = self.trace_contexts[module_type]
            ctx.exit()
            self._log_event("backward_hook_pos", module_type)

        return hook
