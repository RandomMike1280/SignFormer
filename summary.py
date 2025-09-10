import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Dict, List
import rich
from rich.table import Table
from rich.console import Console
from rich.text import Text
import math

# Helper function to format large numbers (like Keras summary)
def format_params(num_params: int) -> str:
    """Formats parameter count into human-readable string (K, M, B)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.3f} B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.3f} M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.3f} K"
    else:
        return str(num_params)

# Helper function to format memory size
def format_size(num_bytes: int) -> str:
    """Formats bytes into human-readable string (B, KB, MB, GB)."""
    if num_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(num_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(num_bytes / p, 2)
    return f"{s} {size_name[i]}"

class ModelSummary:
    """
    Generates a summary of a PyTorch model similar to Keras' model.summary().

    Args:
        model (nn.Module): The PyTorch model to summarize.
        input_size (Optional[Tuple[int, ...]]): The input shape (excluding batch dimension)
            required to perform a forward pass for shape inference.
            E.g., (3, 224, 224) for an image model.
            If None, output shapes will not be calculated.
        dtypes (Optional[List[torch.dtype]]): List of datatypes of input tensors.
            Defaults to [torch.float32].
        device (Optional[torch.device or str]): The device to run the forward pass on.
            Defaults to the model's first parameter's device or 'cpu'.
        print_summary (bool): Whether to print the summary immediately upon initialization.
    """
    def __init__(
        self,
        model: nn.Module,
        input_size: Optional[Tuple[int, ...]] = None,
        dtypes: Optional[List[torch.dtype]] = None,
        device: Optional[torch.device | str] = None,
        print_summary: bool = True,
        detailed: bool = False
    ):
        self.model = model
        self.input_size = input_size
        self.dtypes = dtypes if dtypes is not None else [torch.float32]
        self._hooks = []
        self._output_shapes: Dict[str, Any] = {}
        self._param_counts: Dict[str, int] = {}
        self._layer_info: List[Dict[str, Any]] = []
        self._total_params = 0
        self._trainable_params = 0
        self._model_size_bytes = 0
        self.detailed = detailed

        # Determine device
        if device:
             self.device = torch.device(device)
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration: # Model has no parameters
                self.device = torch.device('cpu')

        # Run analysis
        self._analyze_model()

        if print_summary:
            self.display_summary()

    def _register_hook(self, module: nn.Module, name: str):
        """Registers a forward hook to capture output shape."""
        def hook(mod, inp, outp):
            # Store output shape, handle tuples/lists if necessary
            if isinstance(outp, (list, tuple)):
                 self._output_shapes[name] = [str(tuple(o.shape)) if isinstance(o, torch.Tensor) else type(o).__name__ for o in outp]
            elif isinstance(outp, torch.Tensor):
                 self._output_shapes[name] = str(tuple(outp.shape))
            else:
                 # Handle other potential return types if needed
                 self._output_shapes[name] = type(outp).__name__

        if not isinstance(module, nn.ModuleList) and \
           not isinstance(module, nn.Sequential) and \
           module is not self.model: # Avoid hooking self? maybe needed for complex models
             handle = module.register_forward_hook(hook)
             self._hooks.append(handle)


    def _remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _analyze_model(self):
        """Performs the analysis by iterating through modules and running a forward pass."""

        # --- 1. Iterate through modules to get structure and params ---
        if self.detailed:
            m = self.model.named_modules()
        else:
            m = self.model.named_children()
        for name, module in m:
            # Count parameters for this specific module (and its submodules)
            module_params = sum(p.numel() for p in module.parameters())
            self._param_counts[name] = module_params

            # Register hook only if we need output shapes
            if self.input_size:
                self._register_hook(module, name)

            self._layer_info.append({
                "name": name,
                "type": module.__class__.__name__,
                "params": module_params,
                "output_shape": "N/A" # Placeholder
            })

        # --- 2. Run forward pass for output shapes (if input_size provided) ---
        if self.input_size:
            try:
                # Prepare dummy input
                # Add batch dimension (use 1 or 2 for robustness with BatchNorm)
                batch_size = 2
                dummy_input = [torch.randn(batch_size, *self.input_size, dtype=dtype, device=self.device)
                              for dtype in self.dtypes]
                if len(dummy_input) == 1:
                    dummy_input = dummy_input[0] # Pass single tensor directly if only one input

                # Run forward pass
                self.model.eval() # Set to evaluation mode
                with torch.no_grad():
                    self.model(dummy_input)

            except Exception as e:
                print(f"\n[Warning] Forward pass failed: {e}")
                print("Output shapes could not be determined.")
                # Clear any potentially stored shapes if pass failed midway
                self._output_shapes.clear()
            finally:
                # Always remove hooks
                 self._remove_hooks()


            # Update layer info with captured shapes
            for layer in self._layer_info:
                name = layer["name"]
                if name in self._output_shapes:
                    # Replace batch dim placeholder if needed (often None in Keras)
                    raw_shape = self._output_shapes[name]
                    if isinstance(raw_shape, str) and raw_shape.startswith('(2,'):
                        shape_str = '(None,' + raw_shape[3:]
                    elif isinstance(raw_shape, list):
                        shape_str = '[' + ', '.join(
                             '(None,' + s[3:] if isinstance(s, str) and s.startswith('(2,') else str(s)
                             for s in raw_shape
                        ) + ']'
                    else:
                       shape_str = str(raw_shape) # Fallback
                    layer["output_shape"] = shape_str
                else:
                     # Could happen for modules that don't get called or have no output
                     layer["output_shape"] = "Error/NotCalled?"


        # --- 3. Calculate total parameters and size ---
        param_size_bytes = 0
        for param in self.model.parameters():
            num_params = param.numel()
            self._total_params += num_params
            if param.requires_grad:
                self._trainable_params += num_params
            # Estimate size based on dtype
            param_size_bytes += num_params * param.element_size()

        self._model_size_bytes = param_size_bytes


    def display_summary(self):
        """Prints the formatted summary table to the console."""
        console = Console()

        # --- Model Header ---
        model_name = getattr(self.model, 'name', self.model.__class__.__name__)
        console.print(f"[bold]Model Summary:[/]")
        console.print(f"Model: \"{model_name}\"")

        # --- Table Definition ---
        table = Table(show_header=True, header_style="bold cyan", box=rich.box.HEAVY_EDGE)
        table.add_column("Layer (type)", style="dim", width=35)
        table.add_column("Output Shape", justify="left", style="green", width=25)
        table.add_column("Param #", justify="right", style="magenta", width=15)
        table.add_column("Size", justify="right", style="yellow", width=15)

        # --- Add 'Input' Row Conceptually ---
        input_shape_str = f"(None, {', '.join(map(str, self.input_size))})" if self.input_size else "N/A"
        table.add_row(
             Text(f"{model_name}_input (InputLayer)", style="blue"),
             input_shape_str,
             "0",
             "0 B"
         )
        table.add_section() # Separator after input

        # --- Add Layer Rows ---
        for layer in self._layer_info:
             layer_name = layer['name']
             layer_type = layer['type']
             output_shape = layer['output_shape']
             params_count = layer['params']
             # Estimate size: Assume params are float32 if no specific info
             # A more accurate calculation happens with total size later
             layer_size_bytes = params_count * 4 # Assuming float32 default for individual layers display

             table.add_row(
                 f"{layer_name} ({layer_type})",
                 str(output_shape),
                 format_params(params_count),
                 format_size(layer_size_bytes)
             )

        # --- Footer with Totals ---
        console.print(table) # Print table content first

        total_params_str = format_params(self._total_params)
        trainable_params_str = format_params(self._trainable_params)
        non_trainable_params = self._total_params - self._trainable_params
        non_trainable_params_str = format_params(non_trainable_params)
        total_size_str = format_size(self._model_size_bytes)
        # Estimate input size (assuming float32)
        input_bytes = 0
        if self.input_size:
            input_elements = 1
            for dim in self.input_size:
                input_elements *= dim
            # Assuming float32 for input size calculation display, adjust if needed
            input_bytes = input_elements * 4 * len(self.dtypes) # Batch size is not included in summary size

        estimated_total_size_bytes = self._model_size_bytes + input_bytes
        estimated_total_size_str = format_size(estimated_total_size_bytes)


        console.print(f"Total params: [green]{self._total_params:,}[/] ({total_params_str})")
        console.print(f"Trainable params: [green]{self._trainable_params:,}[/] ({trainable_params_str})")
        console.print(f"Non-trainable params: [yellow]{non_trainable_params:,}[/] ({non_trainable_params_str})")
        console.print("-" * 80)
        console.print(f"Model parameter size: [cyan]{total_size_str}[/]")
        # console.print(f"Estimated Total Size (params + input): [cyan]{estimated_total_size_str}[/]") # Optional: Include input size estimate
        console.print("-" * 80)


# --- Example Usage ---
if __name__ == "__main__":

    # Define a simple example model (similar structure to the image)
    class Encoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, latent_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, latent_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, output_dim)
            self.sigmoid = nn.Sigmoid() # Common for autoencoders reconstructing images/normalized data

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    class Autoencoder(nn.Module):
        def __init__(self, input_dim=784, latent_dim=2):
            super().__init__()
            # Give the submodules explicit names
            self.encoder = Encoder(input_dim, latent_dim)
            self.decoder = Decoder(latent_dim, input_dim)
            self.name = "autoencoder" # Set a custom name attribute

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    # Instantiate the model
    input_dimension = 784 # e.g., MNIST image size (28*28)
    latent_dimension = 2
    model = Autoencoder(input_dim=input_dimension, latent_dim=latent_dimension)

    # Generate and print the summary
    print("--- Basic Example ---")
    # Provide input_size without batch dimension
    summary = ModelSummary(model, input_size=(input_dimension,))

    # --- Example with more layers ---
    print("\n--- More Complex Example ---")
    complex_model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 512), # Assuming input 3x32x32 -> 64x8x8
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    complex_model.name = "CNN_Classifier"
    summary_cnn = ModelSummary(complex_model, input_size=(3, 32, 32))

    # --- Example without input size (no shapes) ---
    print("\n--- Example Without Input Size ---")
    summary_no_shape = ModelSummary(model, input_size=None)