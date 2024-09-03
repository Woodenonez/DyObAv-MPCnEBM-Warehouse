from dataclasses import dataclass
from typing import Any, Union

import casadi as ca # type: ignore

from .mpc_helper import *


@dataclass
class CostTerms:
    cost_rpd: ca.SX = cs.SX(0.0)
    cost_rvd: ca.SX = cs.SX(0.0)
    cost_rtd: ca.SX = cs.SX(0.0)
    cost_input: ca.SX = cs.SX(0.0)
    cost_fleet: ca.SX = cs.SX(0.0)
    cost_fleet_pred: ca.SX = cs.SX(0.0)
    cost_stcobs: ca.SX = cs.SX(0.0)
    cost_dynobs: ca.SX = cs.SX(0.0)
    cost_dynobs_pred: ca.SX = cs.SX(0.0)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if not hasattr(self, __name):
            raise AttributeError(f"Attribute {__name} is not allowed to be set.")
        super().__setattr__(__name, __value)

    def __add__(self, other: "CostTerms") -> "CostTerms":
        added = CostTerms()
        for key in self.__dict__:
            added.__dict__[key] = self.__dict__[key] + other.__dict__[key]
        return added
    
    def sum(self) -> ca.SX:
        return sum(list(self.__dict__.values()))

    def sum_values(self) -> float:
        return float(sum(list(self.__dict__.values())))


def cost_inside_cvx_polygon(state: ca.SX, b: ca.SX, a0: ca.SX, a1: ca.SX, weight:Union[ca.SX, float]=1.0) -> ca.SX:
    """Cost (weighted squared) for being inside a convex polygon defined by `b - [a0,a1]*[x,y]' > 0`.
        
    Args:
        state: The (n*1)-dim target point.
        b:  Shape (1*m) with m half-space offsets.
        a0: Shape (1*m) with m half-space weight vectors.
        a1: Shape (1*m) with m half-space weight vectors.
        
    Returns:
        cost: The (1*1)-dim weighted square cost. If inside, return positive value, else return 0.

    Notes:
        Each half-space if defined as `b - [a0,a1]*[x,y]' > 0`.
        If prod(|max(0,all)|)>0, then the point is inside; Otherwise not.
    """
    indicator = inside_cvx_polygon(state, b, a0, a1)
    cost:ca.SX = weight * (indicator*100)**2 # if the original unit is in meter, then the cost resolution is in cm after *100.
    assert cost.shape == (1,1)
    return cost

def cost_inside_ellipses(state: ca.SX, ellipse_param: list[ca.SX], weight:Union[ca.SX, float]=1.0) -> ca.SX:
    """Cost (weighted squared) for being inside a set of ellipses defined by `(cx, cy, sx, sy, angle, alpha)`.
    
    Args:
        state: The (n*1)-dim target point.
        ellipse_param: Shape (5 or 6 * m) with m ellipses. 
                       Each ellipse is defined by (cx, cy, rx, ry, angle, alpha).
                       
    Returns:
        cost: The (1*m)-dim cost. If inside, return positive value, else return negative value.
    """
    if len(ellipse_param) > 5:
        alpha = ellipse_param[5]
    else:
        alpha = 1
    indicator = inside_ellipses(state, ellipse_param) # indicator<0, if outside ellipse
    indicator = weight * alpha * ca.fmax(0.0, indicator)**2
    cost:ca.SX = ca.sum1(ca.sum2(indicator))
    assert cost.shape == (1,1)
    return cost

def cost_fleet_collision(state: ca.SX, points: ca.SX, safe_distance: float, weight:Union[ca.SX, float]=1.0) -> ca.SX:
    """Cost (weighted squared) for colliding with other robots.
    
    Args:
        state: The (n*1)-dim target point.
        points: The (n*m)-dim points of other robots.
        
    Notes:
        Only have cost when the distance is smaller than `safe_distance`.
    """
    cost:ca.SX = weight * ca.sum2(ca.fmax(0.0, safe_distance**2 - dist_to_points_square(state, points)))
    assert cost.shape == (1,1)
    return cost

def cost_refpath_deviation(state: ca.SX, line_segments: ca.SX, weight:Union[ca.SX, float]=1.0) -> ca.SX:
    """Reference deviation cost (weighted squared) penalizes on the deviation from the reference path.

    Args:
        state: The (n*1)-dim point.
        line_segments: The (n*m)-dim var with m n-dim points.

    Returns:
        The weighted squared distance to the reference path.
    """
    distances_sqrt = ca.SX.ones(1)
    for i in range(line_segments.shape[0]-1):
        distance = dist_to_lineseg(state[:2], line_segments[i:i+2,:2])
        distances_sqrt = ca.horzcat(distances_sqrt, distance**2)
    cost:ca.SX = ca.mmin(distances_sqrt[1:]) * weight
    assert cost.shape == (1,1)
    return cost




