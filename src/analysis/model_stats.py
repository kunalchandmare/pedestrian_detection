from torchinfo import summary, ModelStatistics
import torch
from torch import nn

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