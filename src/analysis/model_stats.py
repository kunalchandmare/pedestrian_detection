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



def inspect_model(name, model):
    weights = [
        param for param_name, param in model.named_parameters()
        if param_name.endswith("weight")
    ]
    state = model.state_dict()

    print(f"\n{name}")
    print("Weight dtypes:", Counter(str(w.dtype) for w in weights))
    print("All parameter dtypes:",
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

def benchmark(model, x, warmup=5, repeats=30):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

        times_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            times_ms.append((time.perf_counter() - start) * 1_000)

    return statistics.median(times_ms)

def inference_time(model_int8, fp32_model, example_inputs):

    x = example_inputs[0].detach().to("cpu", dtype=torch.float32)
    assert tuple(x.shape) == (1, 3, 640, 640)

    # fp32_model is the original, non-exported YOLO DetectionModel.
    fp32_model = fp32_model.cpu().eval()
    model_int8 = model_int8.cpu()  # Do NOT call ordinary .eval() on an exported model.

    fp32_compiled = torch.compile(fp32_model, backend="inductor")
    int8_compiled = torch.compile(model_int8, backend="inductor")


    fp32_ms = benchmark(fp32_compiled, x)
    int8_ms = benchmark(int8_compiled, x)

    print(f"Compiled FP32: {fp32_ms:.2f} ms/image")
    print(f"Compiled INT8: {int8_ms:.2f} ms/image")
    print(f"Speedup: {fp32_ms / int8_ms:.2f}x")