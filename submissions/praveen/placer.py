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
from tqdm import tqdm

from macro_place.benchmark import Benchmark

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

class PraveenPlacer:
    """
    Pure gradient-based optimizer placing. 
    
    Uses the Adam algorithm,
    minimizing wirelength, penalzing overlaps with IoU.
    """

    gap: float = 0.001
    device: torch.Device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_iters: int = 2048

    def _wirelength_cost(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        '''
            Approximate the wirelength cost
        '''
        power = 2
        max_net_length = max([len(net) for net in benchmark.net_nodes])
        net_nodes = [torch.cat((net, torch.full((max_net_length-len(net),), -1, dtype=net.dtype, device=net.device))) for net in benchmark.net_nodes]
        net_nodes = torch.stack(net_nodes, 0)
        net_nodes = torch.where(torch.lt(net_nodes, benchmark.num_macros), net_nodes, -1)

        x_min = torch.where(net_nodes > -1, placement[net_nodes, 0], 1e6)
        x_min = 1 / ((1/x_min).pow(power).sum(1).pow(1/power))
        x_max = torch.where(net_nodes > -1, placement[net_nodes, 0], 0)
        x_max = ((x_max).pow(power).sum(1).pow(1/power))
        y_min = torch.where(net_nodes > -1, placement[net_nodes, 1], 1e6)
        y_min = 1 / ((1/y_min).pow(power).sum(1).pow(1/power))
        y_max = torch.where(net_nodes > -1, placement[net_nodes, 1], 0)
        y_max = ((y_max).pow(power).sum(1).pow(1/power))

        net_hpwl = (x_max - x_min) + (y_max - y_min)
        hpwl = (net_hpwl * benchmark.net_weights).sum()
        net_count = benchmark.net_weights.sum()

        # hpwl = 0
        # net_count = 0
        # power = 2
        # for (net, weight) in zip(benchmark.net_nodes, benchmark.net_weights):
        #     if torch.any(torch.ge(net, benchmark.num_macros)) or torch.any(torch.lt(net, 0)):
        #         continue

        #     # x_min = torch.min(placement[net, 0])
        #     # x_max = torch.max(placement[net, 0])
        #     # y_min = torch.min(placement[net, 1])
        #     # y_max = torch.max(placement[net, 1])
        #     x_min = 1 / ((1/placement[net, 0]).pow(power).sum().pow(1/power))
        #     x_max = ((placement[net, 0]).pow(power).sum().pow(1/power))
        #     y_min = 1 / ((1/placement[net, 1]).pow(power).sum().pow(1/power))
        #     y_max = ((placement[net, 0]).pow(power).sum().pow(1/power))
        #     net_hpwl = x_max - x_min + y_max - y_min

        #     hpwl = hpwl + weight * net_hpwl
        #     net_count = net_count + weight

        cost = hpwl / (net_count * (benchmark.canvas_width + benchmark.canvas_height))

        return cost
    
    def _density_cost(self, placement: torch.Tensor, benchmark: Benchmark, mask: torch.Tensor = None):

        box_list = torch.cat((placement, benchmark.macro_sizes + self.gap), 1)
        if mask is not None:
            box_list = box_list[mask, :]
        iou = box_iou(box_list, box_list, 'cxcywh')

        center_distance_x = box_list[:,0].unsqueeze(0) - box_list[:,0].unsqueeze(1)
        center_distance_y = box_list[:,1].unsqueeze(0) - box_list[:,1].unsqueeze(1)
        center_distance = center_distance_x.abs() + center_distance_y.abs()
        center_distance = torch.exp(-center_distance)

        cost = (iou - torch.eye(len(iou), dtype=iou.dtype, device=iou.device)).abs().mul(center_distance).sum() * 0.5

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
        power = 2
        max_net_length = max([len(net) for net in benchmark.net_nodes])
        net_nodes = [torch.cat((net, torch.full((max_net_length-len(net),), -1, dtype=net.dtype, device=net.device))) for net in benchmark.net_nodes]
        net_nodes = torch.stack(net_nodes, 0)
        net_nodes = torch.where(torch.lt(net_nodes, benchmark.num_macros), net_nodes, -1)

        x_min = torch.where(net_nodes > -1, placement[net_nodes, 0], 1e6)
        x_min = 1 / ((1/x_min).pow(power).sum(1).pow(1/power))
        x_max = torch.where(net_nodes > -1, placement[net_nodes, 0], 0)
        x_max = ((x_max).pow(power).sum(1).pow(1/power))
        y_min = torch.where(net_nodes > -1, placement[net_nodes, 1], 1e6)
        y_min = 1 / ((1/y_min).pow(power).sum(1).pow(1/power))
        y_max = torch.where(net_nodes > -1, placement[net_nodes, 1], 0)
        y_max = ((y_max).pow(power).sum(1).pow(1/power))

        congestion = benchmark.hroutes_per_micron*(x_max - x_min) + benchmark.vroutes_per_micron*(y_max - y_min)
        cost = (congestion * benchmark.net_weights).sum()

        return cost

    def _legalize(self, placement: torch.Tensor, benchmark: Benchmark, movable: torch.Tensor):
        sizes = benchmark.macro_sizes
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        cw = benchmark.canvas_width
        ch = benchmark.canvas_height
        n = len(placement)

        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = torch.zeros(n, dtype=bool, device=self.device)
        legal = placement.clone()
        for idx in tqdm(order, 'Legalizing...'):
            if not movable[idx]:
                placed[idx] = True; continue
            if placed.any():
                dx = torch.abs(legal[idx, 0] - legal[:, 0])
                dy = torch.abs(legal[idx, 1] - legal[:, 1])
                c = (dx < sep_x[idx]+self.gap) & (dy < sep_y[idx]+self.gap) & placed
                c[idx] = False
                if not c.any():
                    placed[idx] = True; continue
            step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
            best_p = legal[idx].clone(); best_d = float('inf')
            for r in range(1, 150):
                found = False
                for dxm in range(-r, r+1):
                    for dym in range(-r, r+1):
                        if abs(dxm) != r and abs(dym) != r: continue
                        cx = torch.clip(placement[idx, 0]+dxm*step, half_w[idx], cw-half_w[idx])
                        cy = torch.clip(placement[idx, 1]+dym*step, half_h[idx], ch-half_h[idx])
                        if placed.any():
                            dx = torch.abs(cx-legal[:, 0]); dy = torch.abs(cy-legal[:, 1])
                            c = (dx < sep_x[idx]+self.gap) & (dy < sep_y[idx]+self.gap) & placed
                            c[idx] = False
                            if c.any(): continue
                        d = (cx-placement[idx, 0])**2+(cy-placement[idx, 1])**2
                        if d < best_d:
                            best_d = d; best_p = torch.tensor([cx, cy], dtype=placement.dtype, device=placement.device); found = True
                if found: break
            legal[idx] = best_p; placed[idx] = True
        return legal

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        benchmark_d = Benchmark_to_device(benchmark, self.device)
        placement = benchmark_d.macro_positions.clone()

        # Only place hard macros; soft macros stay at initial positions
        movable_d = benchmark.get_movable_mask().to(self.device)
        # movable_d = torch.logical_and(benchmark.get_movable_mask(), benchmark.get_hard_macro_mask()).to(self.device)
        hard_macro_mask_d = torch.logical_and(benchmark.get_movable_mask(), benchmark.get_hard_macro_mask()).to(self.device)

        # Mask placement
        free_placement = torch.zeros_like(placement).detach().clone()
        free_placement[movable_d, :] = 1
        fixed_pos = torch.zeros_like(placement)
        fixed_pos[free_placement == 0] = placement.detach().clone()[free_placement == 0]

        placement = placement.requires_grad_()
        optimizer = torch.optim.Adam([placement], lr = 0.01)
        
        for epoch in tqdm(range(self.max_iters), 'Optimizing...'):
            x = placement * free_placement + fixed_pos

            loss = 0
            loss = loss + self._wirelength_cost(x, benchmark_d) # Wirelength
            loss = loss + 10. * self._bounding_cost(x, benchmark_d) # Stay within canvas
            loss = loss + 0.5 * self._density_cost(x, benchmark_d) # Overall density
            loss = loss + 10. * self._density_cost(x, benchmark_d, hard_macro_mask_d) # No hard macro overlap

            loss.backward()
            # print(epoch, loss.item())
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            placement = self._legalize(placement, benchmark_d, hard_macro_mask_d)

        return placement.detach().clone().cpu()
