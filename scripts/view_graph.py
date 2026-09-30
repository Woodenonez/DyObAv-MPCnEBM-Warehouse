"""View the warehouse graph, optionally over its background image."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basic_map.graph import NetGraph
    from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

@dataclass
class ViewArgs:
    background: bool = False


def read_graph_json(json_path: Path) -> NetGraph:
    from basic_map.graph import NetGraph

    with json_path.open("r") as jf:
        data = json.load(jf)
    node_dict = data["node_dict"]
    edge_list = data["edge_list"]
    return NetGraph(node_dict, edge_list)


def main() -> None:
    import tyro

    args: ViewArgs = tyro.cli(ViewArgs)
    import matplotlib.pyplot as plt

    data_dir: Path = ROOT / "data" / "warehouse_sim_original"
    graph: NetGraph = read_graph_json(data_dir / "graph.json")

    fig, ax = plt.subplots()
    ax: Axes
    if args.background:
        ax.imshow(plt.imread(data_dir / "background.png"))
        graph.plot_graph(ax, node_coord=True)
    else:
        graph.plot_graph(ax)
    ax.axis("equal")

    plt.show()


if __name__ == "__main__":
    main()
