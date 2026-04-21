"""
Praveen Placer

A simple but legal placer that:
1. Sorts macros by height (tallest first)
2. Places them left-to-right in rows (like shelf packing)
3. Guarantees zero overlaps and canvas boundary compliance

This produces valid, scoreable placements but makes no attempt to
optimize wirelength, density, or congestion. Use it as a starting
point for your own algorithm.

Usage:
    uv run evaluate submissions/examples/greedy_row_placer.py
    uv run evaluate submissions/examples/greedy_row_placer.py --all
    uv run evaluate submissions/examples/greedy_row_placer.py -b ibm03
"""

import torch
from copy import deepcopy

from macro_place.benchmark import Benchmark
from macro_place.utils import validate_placement

def Benchmark_to_device(benchmark: Benchmark, device: torch.Device) -> Benchmark:

    benchmark_copy = deepcopy(benchmark)

    benchmark_copy.macro_positions = benchmark_copy.macro_positions.to(device)
    benchmark_copy.macro_sizes = benchmark_copy.macro_sizes.to(device)
    benchmark_copy.macro_fixed = benchmark_copy.macro_fixed.to(device)

    for idx in range(benchmark_copy.num_nets):
        benchmark_copy.net_nodes[idx] = benchmark_copy.net_nodes[idx].to(device)
    benchmark_copy.net_weights = benchmark_copy.net_weights.to(device)

    return benchmark_copy

class PraveenPlacer:
    """
    Greedy row-based (shelf packing) placement.

    Places macros in rows from bottom to top, left to right,
    sorted by descending height. Guarantees zero overlaps.
    """

    # Small gap to avoid float32 touching-edge false overlaps
    gap: float = 0.001
    device: torch.Device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _wirelength_cost(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        '''
            Approximate the wirelength cost
        '''
        hpwl = 0
        net_count = 0
        for (net, weight) in zip(benchmark.net_nodes, benchmark.net_weights):
            if torch.any(torch.ge(net, benchmark.num_macros)) or torch.any(torch.lt(net, 0)):
                continue

            x_min = torch.min(placement[net, 0])
            x_max = torch.max(placement[net, 0])
            y_min = torch.min(placement[net, 1])
            y_max = torch.max(placement[net, 1])
            net_hpwl = x_max - x_min + y_max - y_min

            hpwl = hpwl + weight * net_hpwl
            net_count = net_count + weight

        cost = hpwl / (net_count * (benchmark.canvas_width + benchmark.canvas_height))

        return cost

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        benchmark_d = Benchmark_to_device(benchmark, self.device)
        placement = benchmark_d.macro_positions.clone()
        # Only place hard macros; soft macros stay at initial positions
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()

        sizes = benchmark.macro_sizes
        sizes = sizes.to(self.device)
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        # Sort movable macros by height descending (shelf packing heuristic)
        movable_indices.sort(key=lambda i: -sizes[i, 1].item())

        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0

        for idx in movable_indices:
            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()

            # Start new row if macro doesn't fit
            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_height + self.gap
                row_height = 0.0

            # Check if we've run out of vertical space
            if cursor_y + h > canvas_h:
                # Place at origin as fallback (will overlap but shouldn't happen
                # if area utilization < 100%)
                placement[idx, 0] = w / 2
                placement[idx, 1] = h / 2
                continue

            # Place macro (positions are centers)
            placement[idx, 0] = cursor_x + w / 2
            placement[idx, 1] = cursor_y + h / 2

            cursor_x += w + self.gap
            row_height = max(row_height, h)

        wirelength_cost = self._wirelength_cost(placement, benchmark_d)
        print(wirelength_cost)

        return placement.cpu()
