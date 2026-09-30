import statistics
import time

from torchinfo import summary, ModelStatistics
import torch
from torch import nn
from collections import Counter

def calculate_sparsity(model):
    zero_params = 0
    total_params = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.detach()

            zero_params += torch.sum(weight == 0).item()
            total_params += weight.numel()

    return zero_params / total_params

def params_stats(model, example_input_size, verbose=1 ) -> list[tuple[str, str]]:
    """
    Print only the parameter summary from a ModelStatistics object.
    """
    sparsity = calculate_sparsity(model)

    stats = summary(
        model,
        #input_size=(1, 3, 320, 320),
        input_size=example_input_size,
        col_names=("input_size", "output_size", "num_params")
    )

    # Recreate the same formatting logic as in ModelStatistics.__repr__
    formatting = stats.formatting

    total_params = ModelStatistics.format_output_num(
        stats.total_params, formatting.params_units
    )
    trainable_params = ModelStatistics.format_output_num(
        stats.trainable_params, formatting.params_units
    )
    non_trainable_params = ModelStatistics.format_output_num(
        stats.total_params - stats.trainable_params,
        formatting.params_units,
    )

    macs = ModelStatistics.format_output_num(
        stats.total_mult_adds, formatting.macs_units
    )
    input_size = stats.to_megabytes(stats.total_input)
    output_bytes = stats.to_megabytes(stats.total_output_bytes)
    param_bytes = stats.to_megabytes(stats.total_param_bytes)
    total_bytes = stats.to_megabytes(
        stats.total_input + stats.total_output_bytes + stats.total_param_bytes
    )

    # Build a simple table
    rows = [
        ("Sparsity(%)", f"{sparsity * 100:.2f}"),
        ("Total params", total_params.split(": ")[1]),
        ("Trainable params", trainable_params.split(": ")[1]),
        ("Non-trainable params", non_trainable_params.split(": ")[1]),
        ("Total mult-adds (Units.MEGABYTES)", macs.split(": ")[1]),
        ("Input size (MB)", f"{input_size:.2f}"),
        ("Forward/backward pass size (MB)", f"{output_bytes:.2f}"),
        ("Params size (MB)", f"{param_bytes:.2f}"),
        ("Estimated Total Size (MB)", f"{total_bytes:.2f}"),
    ]

    if verbose == 1:
        name_width = max(len(name) for name, _ in rows)
        value_width = max(len(value) for _, value in rows)
        print("=" * (name_width + value_width + 5))
        print(f"{'Metric':<{name_width}} | {'Value':>{value_width}}")
        print("=" * (name_width + value_width + 5))

        for name, value in rows:
            print(f"{name:<{name_width}} | {value:>{value_width}}")

        print("=" * (name_width + value_width + 5))

    return rows



def quant_inspect_model(name, model):
    """
              ┌── DQ₁ ──→ operation A
FP32 ── Q ────┤
              └── DQ₂ ──→ operation B
    Inspect and analyze the quantization and state of a PyTorch model.

    This function evaluates the properties and state of a given PyTorch model,
    with a focus on assessing quantization-related aspects such as tensor data
    types and the memory usage of model state. It also analyzes the model's
    graph to count and identify quantization operations if the model has a
    graph attribute.
    """
    weights = [
        param for param_name, param in model.named_parameters()
        if param_name.endswith("weight")
    ]
    state = model.state_dict()

    print(f"\n{name}")
    print("Weight Tensors dtypes:", Counter(str(w.dtype) for w in weights))
    print("Parameter Tensors dtypes:",
          Counter(str(p.dtype) for p in model.parameters()))
    print("Stored state tensor size:",
          round(sum(t.numel() * t.element_size()
                    for t in state.values()
                    if isinstance(t, torch.Tensor)) / 1e6, 2),
          "MB")

    if hasattr(model, "graph"):
        quant_ops = [
            str(node.target)
            for node in model.graph.nodes
                if "quantiz" in str(node.target).lower()
        ]
        print("Quantization ops:", len(quant_ops))
        print("Examples:", quant_ops[:6])

def time_ms(model, x, warmup=5, repeats=30):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

        times_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            times_ms.append((time.perf_counter() - start) * 1_000)

    return statistics.median(times_ms)

def inference_time(model_int8, fp32_model, example_inputs, device="cpu"):

    x = example_inputs[0].detach().to(device, dtype=torch.float32)
    assert tuple(x.shape) == (1, 3, 640, 640)

    # fp32_model is the original, non-exported YOLO DetectionModel.
    fp32_model = fp32_model.cpu().eval()
    model_int8 = model_int8.cpu()  # Do NOT call ordinary .eval() on an exported model.

    #Need C++ compiler with VS Build V19 above 16.11 toolset (currently not available so compile fails)
    #fp32_compiled = torch.compile(fp32_model, backend="inductor")
    #int8_compiled = torch.compile(model_int8, backend="inductor")


    fp32_ms = time_ms(fp32_model, x)
    int8_ms = time_ms(model_int8, x)

    print(f"FP32 Model: {fp32_ms:.2f} ms/image")
    print(f"INT8 Quant Model: {int8_ms:.2f} ms/image")
    print(f"Speedup: {fp32_ms / int8_ms:.2f}x")

def describe_output(output, label):
    print(f"\n{label}: {type(output).__name__}")
    if isinstance(output, torch.Tensor):
        print("  shape:", tuple(output.shape), "dtype:", output.dtype)
    elif isinstance(output, (tuple, list)):
        for i, item in enumerate(output):
            describe_output(item, f"{label}[{i}]")
    elif isinstance(output, dict):
        for key, item in output.items():
            describe_output(item, f"{label}[{key!r}]")
    else:
        print("  value type:", type(output).__name__)


def inference_raw_out(model, example_input, device="cpu"):

    assert tuple(example_input.shape) == (1, 3, 640, 640) # Expected shape: (1, 3, 640, 640)

    #model.eval()

    with torch.inference_mode():
        raw_int8 = model(example_input)

    return raw_int8

def check_outputs(model_int8, fp32_model, example_inputs, device="cpu"):

    x = example_inputs[0] if isinstance(example_inputs, tuple) else example_inputs

    raw_fp32 = inference_raw_out(fp32_model,x)
    raw_int8 = inference_raw_out(model_int8,x)

    describe_output(raw_fp32, "FP32")
    describe_output(raw_int8, "INT8-converted")

    fp = raw_fp32[0]
    q = raw_int8[0]

    print("Shapes:", fp.shape, q.shape)
    print("Mean absolute difference:", (fp - q).abs().mean().item())
    print("Max absolute difference:", (fp - q).abs().max().item())
    print("FP32 finite:", torch.isfinite(fp).all().item())
    print("Converted finite:", torch.isfinite(q).all().item())