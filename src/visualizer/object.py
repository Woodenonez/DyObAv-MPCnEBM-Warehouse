import math
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow


class ObjectVisualizer(metaclass=ABCMeta):
    """Visualizer of the object."""
    def __init__(self, spec):
        self.spec = spec
        self.with_plot = False

    @abstractmethod
    def _get_vis_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def plot(self, *args, **kwargs):
        self.with_plot = True
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        if not self.with_plot:
            raise ValueError('Update is called before plot.')
        pass


class PointSpeedObjectVisualizer(ObjectVisualizer):
    """For an object with a point-model and speed profile."""
    def __init__(self, zero_speed_length: float, max_speed_length: float, max_speed: float):
        self.zero_speed_length = zero_speed_length
        self.max_speed_length = max_speed_length
        self.max_speed = max_speed
        self.with_plot = False

    def _get_vis_data(self, x: float, y: float, yaw: float, speed:float=0.0, angular_speed:float=0.0):
        """Get the data to be visualized.
        
        Args:
            angular_speed: angular speed of the object [rad/s].

        Returns:
            speed_direction: Vector indicating the speed direction.
            turning_direction: Vector indicating the turning direction.
        """
        speed_length =  self.zero_speed_length + (self.max_speed_length-self.zero_speed_length) * min(1, speed/self.max_speed)
        speed_direction = (speed_length*np.cos(yaw), speed_length*np.sin(yaw))
        turning_direction = (abs(angular_speed)*np.cos(yaw + np.sign(angular_speed)*np.deg2rad(90)), abs(angular_speed)*np.sin(yaw + np.sign(angular_speed)*np.deg2rad(90)))
        return speed_direction, turning_direction

    def plot(self, ax: Axes, x: float, y: float, yaw: float, speed:float=0.0, angular_speed:float=0.0, object_color='k', indicator_color='r', **indicator_kwargs) -> None:
        """Plot the point-speed object."""
        speed_direction, turning_direction = self._get_vis_data(x, y, yaw, speed, angular_speed)

        point_0 = ax.plot(x, y, 'o', color=object_color)[0]
        indicator_0 = ax.arrow(x, y, speed_direction[0], speed_direction[1], head_width=0.1, head_length=0.2, fc=object_color, **indicator_kwargs)
        indicator_1 = ax.arrow(x, y, turning_direction[0], turning_direction[1], head_width=0.1, head_length=0.2, fc=indicator_color, **indicator_kwargs)
        self.obj_vis_tuple: tuple[Line2D, FancyArrow, FancyArrow] = (point_0, indicator_0, indicator_1)
        self.with_plot = True

    def update(self, x: float, y: float, yaw: float, speed:float=0.0, angular_speed:float=0.0) -> None:
        """Update the plot of the point-speed object."""
        if not self.with_plot:
            raise ValueError("Plot the object first.")
        
        speed_direction, turning_direction = self._get_vis_data(x, y, yaw, speed, angular_speed)

        self.obj_vis_tuple[0].set_data([x], [y])
        self.obj_vis_tuple[1].set_data(x=x, y=y, dx=speed_direction[0], dy=speed_direction[1])
        self.obj_vis_tuple[2].set_data(x=x, y=y, dx=turning_direction[0], dy=turning_direction[1])


class CircularObjectVisualizer(ObjectVisualizer):
    def __init__(self, radius: float, indicate_angle:bool=True):
        self.radius = radius
        self.indicate_angle = indicate_angle
        self.with_plot = False

        self.moving_patches: list[Circle] = []

    def _get_vis_data(self, x: float, y: float, yaw: Optional[float]) -> Optional[tuple[float, float]]:
        direction = None
        if (yaw is not None) and self.indicate_angle:
            direction = (float(self.radius*np.cos(yaw)), float(self.radius*np.sin(yaw)))
        return direction
    
    def plot(self, ax: Axes, x: float, y: float, yaw:Optional[float]=None, object_color='k', indicator_color='r', **indicator_kwargs) -> None:
        """Plot the circular object.

        Args:
            yaw: yaw angle of the object [rad]
            object_color: color of the object
            indicator_color: color of the indicator (triangle)
        """
        if yaw is None:
            yaw = 0.0
        direction = self._get_vis_data(x, y, yaw)

        # Plot the object
        obj = Circle((x, y), radius=self.radius, color=object_color)
        self.moving_patches.append(obj)
        ax.add_patch(obj)

        # Plot the indicator
        if self.indicate_angle:
            assert direction is not None
            indicator = ax.arrow(x, y, direction[0], direction[1], head_width=0.1, head_length=0.2, fc=indicator_color, **indicator_kwargs)
            self.obj_vis_tuple = [indicator]

        self.with_plot = True

    def update(self, x: float, y: float, yaw:Optional[float]=None) -> None:
        """Update the plot of the circular object."""
        if not self.with_plot:
            raise ValueError("Plot the object first.")
        
        self.moving_patches[0].center = (x, y)
        if self.indicate_angle:
            direction = self._get_vis_data(x, y, yaw)
            assert direction is not None
            self.obj_vis_tuple[0].set_data(x=x, y=y, dx=direction[0], dy=direction[1])


class FourWheeledObjectSpecification:
    """Object specification for four-wheeled ones.

    Args:
        length: length of the object [m]
        width: width of the object [m]
        back_to_wheel: distance from the back of the object to the center of the rear wheels [m]
        wheel_len: length of the wheel [m]
        wheel_width: width of the wheel [m]
        tread: distance between the two wheels [m]
        wb: wheelbase (distance between the front and rear wheels) [m]
    """
    def __init__(self, length: float, width: float, back_to_wheel: float, wheel_len: float, wheel_width: float, tread: float, wb: float):
        self.length = length
        self.width = width
        self.back_to_wheel = back_to_wheel
        self.wheel_len = wheel_len
        self.wheel_width = wheel_width
        self.tread = tread
        self.wb = wb

class FourWheeledObjectVisualizer(ObjectVisualizer):
    def __init__(self, spec: FourWheeledObjectSpecification):
        self.spec = spec
        self.with_plot = False

        self.moving_patches: list[Circle] = []

    def _get_vis_data(self, x: float, y: float, yaw: float, steer:float=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
        """Get visualization data of the object.

        Args:
            yaw: yaw angle [rad]
            steer: steering angle to the body frame [rad]
        
        Returns:
            outline: outline of the object [2x5]
            fr_wheel: front right wheel [2x5]
            fl_wheel: front left wheel [2x5]
            rr_wheel: rear right wheel [2x5]
            rl_wheel: rear left wheel [2x5]
            circumcircle: circumcircle of the object (center x, center y, radius) [3]
        """
        l, w = self.spec.length, self.spec.width
        b2w = self.spec.back_to_wheel
        wl, ww = self.spec.wheel_len, self.spec.wheel_width
        t = self.spec.tread
        wb = self.spec.wb

        outline = np.array([[-b2w, (l-b2w), (l-b2w), -b2w, -b2w],
                        [w/2, w/2, - w/2, -w/2, w/2]])

        fr_wheel = np.array([[wl, -wl, -wl, wl, wl],
                            [-ww-t, -ww-t, ww-t, ww-t, -ww-t]])

        rr_wheel = np.copy(fr_wheel)

        fl_wheel = np.copy(fr_wheel)
        fl_wheel[1, :] *= -1
        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
        Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                        [-math.sin(steer), math.cos(steer)]])

        fr_wheel = (fr_wheel.T.dot(Rot2)).T
        fl_wheel = (fl_wheel.T.dot(Rot2)).T
        fr_wheel[0, :] += wb
        fl_wheel[0, :] += wb

        fr_wheel = (fr_wheel.T.dot(Rot1)).T
        fl_wheel = (fl_wheel.T.dot(Rot1)).T

        outline = (outline.T.dot(Rot1)).T
        rr_wheel = (rr_wheel.T.dot(Rot1)).T
        rl_wheel = (rl_wheel.T.dot(Rot1)).T

        translate = np.array([[x], [y]])

        outline += translate
        fr_wheel += translate
        rr_wheel += translate
        fl_wheel += translate
        rl_wheel += translate

        cx = (outline[0, 0] + outline[0, 2]) / 2.0
        cy = (outline[1, 0] + outline[1, 2]) / 2.0
        r = math.hypot(cx - outline[0, 0], cy - outline[1, 0])

        return outline, fr_wheel, rr_wheel, fl_wheel, rl_wheel, (cx, cy, r)

    def plot(self, ax: Axes, x: float, y: float, yaw: float, steer:float=0.0, object_color="k", circle_color="r", highlight_color:Optional[str]=None) -> None:
        """Plotting a object mimicing a real car.

        Arguments:
            yaw: yaw angle [rad]
            steer: steering angle to the body frame [rad]
            object_color: color of the body of the object
            circle_color: color of the circumcircle of the object
            highlight_color: color of the front wheels
        """
        if highlight_color is None:
            highlight_color = object_color

        ol, fr, rr, fl, rl, (cx, cy, r) = self._get_vis_data(x, y, yaw, steer)

        line_0 = ax.plot(ol[0, :], ol[1, :], object_color)[0]
        line_1 = ax.plot(fr[0, :], fr[1, :], highlight_color)[0]
        line_2 = ax.plot(rr[0, :], rr[1, :], object_color)[0]
        line_3 = ax.plot(fl[0, :], fl[1, :], highlight_color)[0]
        line_4 = ax.plot(rl[0, :], rl[1, :], object_color)[0]
        point_0 = ax.plot(x, y, "k*")[0]

        circle_0 = Circle(xy=(cx, cy), radius=r, fill=False, color=circle_color, ls='--')
        self.moving_patches.append(circle_0)
        ax.add_patch(circle_0)

        self.obj_vis_tuple = [line_0, line_1, line_2, line_3, line_4, point_0]
        self.with_plot = True

    def update(self, x: float, y: float, yaw: float, steer:float=0.0) -> None:
        """Update the object visualization."""
        if not self.with_plot:
            raise ValueError("Plot the object first.")

        ol, fr, rr, fl, rl, (cx, cy, _) = self._get_vis_data(x, y, yaw, steer)

        self.obj_vis_tuple[0].set_data(ol[0, :], ol[1, :])
        self.obj_vis_tuple[1].set_data(fr[0, :], fr[1, :])
        self.obj_vis_tuple[2].set_data(rr[0, :], rr[1, :])
        self.obj_vis_tuple[3].set_data(fl[0, :], fl[1, :])
        self.obj_vis_tuple[4].set_data(rl[0, :], rl[1, :])
        self.obj_vis_tuple[5].set_data(x, y)

        self.moving_patches[0].center = (cx, cy)
        


if __name__ == '__main__':

    boundary = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]

    # Vehicle parameters
    LENGTH = 4.5  # [m]
    WIDTH = 2.0  # [m]
    BACKTOWHEEL = 1.0  # [m]
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.7  # [m]
    WB = 2.5  # [m]

    spec_1 = FourWheeledObjectSpecification(LENGTH, WIDTH, BACKTOWHEEL, WHEEL_LEN, WHEEL_WIDTH, TREAD, WB)
    viser_1 = FourWheeledObjectVisualizer(spec_1)

    viser_2 = CircularObjectVisualizer(WIDTH/2, indicate_angle=True)

    viser_3 = PointSpeedObjectVisualizer(0.5, 1, 2)

    viser = viser_3

    xs = np.linspace(2, 5, 20)
    ys = np.linspace(2, 5, 20)
    yaws = np.linspace(0, 90, 20)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.grid(True)
    ax.axis("equal")
    ax.set_xlim(np.array(boundary)[:, 0].min(), np.array(boundary)[:, 0].max())
    ax.set_ylim(np.array(boundary)[:, 1].min(), np.array(boundary)[:, 1].max())

    viser.plot(ax, x=0, y=0, yaw=np.deg2rad(30), speed=2, angular_speed=-0.5)
    plt.pause(0.1)
    for x, y, yaw in zip(xs, ys, yaws):
        viser.update(x, y, np.deg2rad(yaw), 2, -0.5)
        plt.pause(0.1)
    plt.show()