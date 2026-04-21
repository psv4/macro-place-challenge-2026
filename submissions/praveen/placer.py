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
from torchvision.ops import box_iou, box_convert
from copy import deepcopy
from pathlib import Path

from macro_place.benchmark import Benchmark
from macro_place.utils import validate_placement
from macro_place.objective import compute_proxy_cost, _set_placement

def Benchmark_to_device(benchmark: Benchmark, device: torch.Device) -> Benchmark:

    benchmark_copy = deepcopy(benchmark)

    benchmark_copy.macro_positions = benchmark_copy.macro_positions.to(device)
    benchmark_copy.macro_sizes = benchmark_copy.macro_sizes.to(device)
    benchmark_copy.macro_fixed = benchmark_copy.macro_fixed.to(device)

    for idx in range(benchmark_copy.num_nets):
        benchmark_copy.net_nodes[idx] = benchmark_copy.net_nodes[idx].to(device)
    benchmark_copy.net_weights = benchmark_copy.net_weights.to(device)

    benchmark_copy.port_positions = benchmark_copy.port_positions.to(device)

    for idx in range(len(benchmark_copy.macro_pin_offsets)):
        benchmark_copy.macro_pin_offsets[idx] = benchmark_copy.macro_pin_offsets[idx].to(device)

    return benchmark_copy

def _load_plc(name):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root.as_posix()))
        return plc
    ng45 = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None

def _extract_edges(benchmark, plc):
    n_hard = benchmark.num_hard_macros
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    edge_dict = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            ml = sorted(macros)
            w = 1.0 / (len(ml) - 1)
            for i in range(len(ml)):
                for j in range(i + 1, len(ml)):
                    pair = (ml[i], ml[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + w
    if not edge_dict:
        return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0)
    return (torch.tensor(list(edge_dict.keys()), dtype=torch.long),
            torch.tensor([edge_dict[e] for e in edge_dict], dtype=torch.float32))

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
    
    def _density_cost(self, placement: torch.Tensor, benchmark: Benchmark, mask: torch.Tensor = None):

        box_list = torch.cat((placement, benchmark.macro_sizes + self.gap), 1)
        if mask is not None:
            box_list = box_list[mask, :]
        iou = box_iou(box_list, box_list, 'cxcywh')

        cost = (iou - torch.eye(len(iou)).to(iou)).abs().sum() * 0.5

        return cost
    
    def _bounding_cost(self, placement: torch.Tensor, benchmark: Benchmark):

        box_list = torch.cat((placement, benchmark.macro_sizes + self.gap), 1)
        box_list = box_convert(box_list, 'cxcywh', 'xyxy')

        canvas_box = torch.Tensor([0., 0., benchmark.canvas_width, benchmark.canvas_height]).to(box_list)

        oob_dist = torch.stack(
            (
                canvas_box[0] - box_list[:,0],
                canvas_box[1] - box_list[:,1],
                box_list[:,2] - canvas_box[2],
                box_list[:,3] - canvas_box[3]
            ),
            dim=0
        )

        cost = torch.nn.functional.relu(oob_dist).square().sum()

        return cost
    
    def _congestion_cost(self, placement: torch.Tensor, benchmark: Benchmark):
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
        plc = _load_plc(benchmark.name)
        edges, edge_weights = _extract_edges(benchmark, plc)

        benchmark_d = Benchmark_to_device(benchmark, self.device)
        edges = edges.to(self.device)
        edge_weights = edge_weights.to(self.device)
        placement = benchmark_d.macro_positions.clone()

        # Only place hard macros; soft macros stay at initial positions
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()
        movable_d = movable.to(self.device)

        # Mask placement
        free_placement = torch.zeros_like(placement).detach().clone()
        free_placement[movable_d, :] = 1
        fixed_pos = torch.zeros_like(placement)
        fixed_pos[free_placement == 0] = placement.detach().clone()[free_placement == 0]

        placement = placement.requires_grad_()
        optimizer = torch.optim.Adam([placement], lr = 0.001)
        
        for epoch in range(100):
            x = placement * free_placement + fixed_pos

            loss = 0
            #loss = loss + self._wirelength_cost(x, benchmark_d) # Wirelength
            #loss = loss + 10. * self._bounding_cost(x, benchmark_d) # Stay within canvas
            #loss = loss + 0.5 * self._density_cost(x, benchmark_d) # Overall density
            loss = loss + 10000. * self._density_cost(x, benchmark_d, movable_d) # No hard macro overlap

            loss.backward()
            print(epoch, loss.item())
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                # placement = placement * free_placement + fixed_pos
                print(self._density_cost(x, benchmark_d, movable_d))
                print(compute_proxy_cost(x, benchmark_d, plc))

        return placement.detach().clone().cpu()
