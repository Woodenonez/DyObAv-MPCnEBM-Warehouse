# Basic package: map (MAP)
The package MAP is a basic package for map-related operations. It is mainly responsible for:
- (a) reading, storing, and processing basic map data into geometric map objects or occupancy grid maps; 
- (b) reading and generating Networkx graphs storing predefined map nodes and edges.

(*A basic package is a package that is not dependent on other packages.*)

## Dependencies
- numpy
- matplotlib
- networkx (for graph generation and processing)
- scipy, scikit-image (for image processing in OccupancyMap)

## I/O
- Input: JSON file containing map/graph data.
- Output: Map/Graph objects containing basic map/graph data.

## Other usage
- Provide visualization functions for map/graph data.
- Provide simple functions for common map/graph data processing.