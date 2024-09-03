from extremitypathfinder.extremitypathfinder import PolygonEnvironment # type: ignore

PathNode = tuple[float, float]


class VisibilityPathFinder:
    """Generate the reference path via the visibility graph and A* algorithm.

    Attributes:
        env: The environment object of solving the visibility graph.
        boundary_coords: The coordinates of the boundary of the environment.
        obstacle_list: The list of obstacles in the environment.

    Methods:
        __prepare: Prepare the visibility graph including preprocess the map.
        update_env: Update the environment with new boundary and obstacles.
        get_ref_path: Get the (shortest) refenence path.
    """
    def __init__(self, boundary_coords: list[PathNode], obstacle_list: list[list[PathNode]], verbose=False):
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        self.vb = verbose
        self.__prepare()

    def __prepare(self):
        self.env = PolygonEnvironment()
        self.env.store(self.boundary_coords, self.obstacle_list) # pass obstacles and boundary to environment
        self.env.prepare() # prepare the visibility graph

    def update_env(self, boundary_coords: list[PathNode], obstacle_list: list[list[PathNode]]):
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        self.__prepare()

    def get_ref_path(self, start_pos: PathNode, end_pos: PathNode) -> list[PathNode]:
        """
        Description:
            Generate the initially guessed path based on obstacles and boundaries specified during preparation.
        Args:
            start_pos: The x,y coordinates.
            end_pos: - The x,y coordinates.
        Returns:
            path: List of coordinates of the inital path
        """
        if self.vb:
            print(f'{self.__class__.__name__} Reference path generated.')

        try:
            path, dist = self.env.find_shortest_path(start_pos[:2], end_pos[:2]) # 'dist' are distances of every segments.
        except Exception as e:
            print(f'{self.__class__.__name__} With start {start_pos} and goal {end_pos}.')
            print(f'{self.__class__.__name__} With boundary {self.boundary_coords}.')
            raise e
        return path


if __name__ == "__main__":
    import matplotlib.pyplot as plt # type: ignore
    import timeit

    boundary_coords = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
    obstacle_list = [[(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 0.1)], 
                     [(3.0, 3.0), (3.0, 4.0), (4.0, 4.0), (4.0, 3.0)],
                     [(4.5, 5.5), (4.5, 7.0), (6.0, 6.0), (6.0, 5.0)],
                     [(7.0, 7.0), (7.0, 8.0), (8.0, 8.0), (8.0, 7.0)]]
    start_pos = (0.5, 0.5)
    end_pos = (9.5, 9.5)

    path_finder_1 = VisibilityPathFinder(boundary_coords, obstacle_list)

    repeat = 500
    start_time = timeit.default_timer()
    for _ in range(repeat):
        path_1 = path_finder_1.get_ref_path(start_pos, end_pos)
    end_time = timeit.default_timer()
    print(f'Cost time for {repeat} runs: {round(end_time-start_time, 4)}s, average: {1000*round((end_time-start_time), 4)/repeat}ms')


    fig, ax = plt.subplots()
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])
    ax.plot([x[0] for x in boundary_coords], [x[1] for x in boundary_coords], 'k')
    for obs in obstacle_list:
        ax.plot([x[0] for x in obs], [x[1] for x in obs], 'k')
    ax.plot([x[0] for x in path_1], [x[1] for x in path_1], 'r--', label='path_1')
    ax.set_aspect('equal')
    plt.legend()
    plt.show()