import math
from typing import Any, Sequence, Tuple, Union

from torch_cfd import grids
from torch_cfd.boundaries import (
    BCType,
    ConstantBoundaryConditions,
    ImmersedBoundaryConditions,
)

velocity_bc_types = Any
pressure_bc_types = Any


def get_inlet_velocities_from_angle(
    angle: float, angle_units: str = "degrees"
) -> Tuple[float, float]:
    """Convert an angle to inlet velocity components with magnitude 1.

    Args:
        angle: Angle of the inflow direction.
               For angle_units="degrees": 0° = right (1, 0), 90° = up (0, 1),
               -90° = down (0, -1), 180° = left (-1, 0)
               For angle_units="radians": 0 = right (1, 0), π/2 = up (0, 1)
        angle_units: Either "degrees" or "radians" (default: "degrees")

    Returns:
        Tuple of (x_inlet_velocity, y_inlet_velocity) with magnitude 1

    Examples:
        >>> get_inlet_velocities_from_angle(0)
        (1.0, 0.0)
        >>> get_inlet_velocities_from_angle(90)
        (0.0, 1.0)
        >>> get_inlet_velocities_from_angle(-45)
        (0.707..., -0.707...)
    """
    if angle_units.lower() == "degrees":
        angle_rad = math.radians(angle)
    elif angle_units.lower() == "radians":
        angle_rad = angle
    else:
        raise ValueError(
            f"angle_units must be 'degrees' or 'radians', got '{angle_units}'"
        )

    x_inlet_velocity = math.cos(angle_rad)
    y_inlet_velocity = math.sin(angle_rad)

    return (x_inlet_velocity, y_inlet_velocity)


def karman_vortex_square_velocity_boundary_conditions(
    grid: grids.Grid,
    inlet_velocity: Tuple[float, float] = (1.0, 0.0),
    square_center: Tuple[float, float] = (0.4, 0.5),
    square_halfwidth: float = 0.05,
    immersed_bc_values: float = 0.0,
    periodic_y: bool = False,
) -> velocity_bc_types:
    """Create separate velocity boundary conditions for u and v components in 2d von Karman vortex street simulation with a square obstacle.

    Args:
        grid: Computational grid
        inlet_velocity: Inlet velocity (u, v) components
        square_center: Center coordinates of square obstacle
        square_halfwidth: Half-width of square obstacle
        immersed_bc_values: Value enforced inside solid regions
        periodic_y: If True, use periodic boundary conditions for upper and lower walls (y-direction)
    """
    if grid.ndim != 2:
        raise ValueError(
            "Karman vortex boundary conditions are only valid for 2D grids"
        )

    # Set y-direction boundary types based on periodic_y
    if periodic_y:
        u_y_type = (BCType.PERIODIC, BCType.PERIODIC)
        v_y_type = (BCType.PERIODIC, BCType.PERIODIC)
        u_y_values = (None, None)  # Periodic BCs have None values
        v_y_values = (None, None)
    else:
        u_y_type = (
            BCType.NEUMANN,
            BCType.NEUMANN,
        )  # y-direction: slip walls (zero normal gradient)
        v_y_type = (
            BCType.DIRICHLET,
            BCType.DIRICHLET,
        )  # y-direction: no penetration at walls
        # y-direction: zero gradient at walls
        u_y_values = (0.0, 0.0)  # type: ignore[assignment]
        # y-direction: zero normal velocity at walls
        v_y_values = (0.0, 0.0)  # type: ignore[assignment]

    # U-velocity boundary conditions
    u_types = (
        (
            BCType.DIRICHLET,
            BCType.NEUMANN,
        ),  # x-direction: inlet Dirichlet, outlet Neumann
        u_y_type,
    )
    u_values = (
        (inlet_velocity[0], 0.0),  # x-direction: inlet u-velocity, zero gradient outlet
        u_y_values,
    )

    # V-velocity boundary conditions
    v_types = (
        (
            BCType.DIRICHLET,
            BCType.NEUMANN,
        ),  # x-direction: inlet Dirichlet, outlet Neumann
        v_y_type,
    )
    v_values = (
        (inlet_velocity[1], 0.0),  # x-direction: inlet v-velocity, zero gradient outlet
        v_y_values,
    )

    u_bc = ImmersedBoundaryConditions(
        types=u_types,
        values=u_values,
        center=(square_center,),
        radius=(square_halfwidth,),
        num_obstacles=1,
        shape="square",
        immersed_bc_value=immersed_bc_values,
        grid=grid,
        offset=(1, 0.5),
    )

    v_bc = ImmersedBoundaryConditions(
        types=v_types,
        values=v_values,
        center=(square_center,),
        radius=(square_halfwidth,),
        num_obstacles=1,
        shape="square",
        immersed_bc_value=immersed_bc_values,
        grid=grid,
        offset=(0.5, 1),
    )

    return u_bc, v_bc


def karman_vortex_square_boundary_conditions(
    grid: grids.Grid,
    inlet_velocity: Tuple[float, float] = (1.0, 0.0),
    inlet_pressure: float = 0.0,
    outlet_pressure: float = 0.0,
    square_center: Tuple[float, float] = (0.4, 0.5),
    square_halfwidth: float = 0.05,
    immersed_bc_value: float = 0.0,
    periodic_y: bool = False,
) -> Tuple[velocity_bc_types, pressure_bc_types]:
    """Create complete set of boundary conditions for Kármán vortex street with a square obstacle.

    Args:
        grid: Computational grid
        inlet_velocity: Inlet velocity (u, v) components
        inlet_pressure: Pressure at inlet
        outlet_pressure: Pressure at outlet
        square_center: Center coordinates of square obstacle
        square_halfwidth: Half-width of square obstacle
        immersed_bc_value: Value enforced inside solid regions
        periodic_y: If True, use periodic boundary conditions for upper and lower walls (y-direction)

    Returns:
        Tuple of (velocity_bc, pressure_bc) boundary conditions
    """
    u_bc, v_bc = karman_vortex_square_velocity_boundary_conditions(
        grid,
        inlet_velocity,
        square_center,
        square_halfwidth,
        immersed_bc_value,
        periodic_y=periodic_y,
    )

    # Set pressure y-direction BC based on periodic_y
    if periodic_y:
        p_y_type = (BCType.PERIODIC, BCType.PERIODIC)
        p_y_values = (None, None)
    else:
        p_y_type = (BCType.NEUMANN, BCType.NEUMANN)
        p_y_values = (0.0, 0.0)  # type: ignore[assignment]

    p_bc = ConstantBoundaryConditions(
        types=(
            (
                BCType.NEUMANN,
                BCType.DIRICHLET,
            ),
            p_y_type,
        ),
        values=((inlet_pressure, outlet_pressure), p_y_values),
    )
    return (u_bc, v_bc), p_bc


def karman_vortex_multiple_squares_velocity_boundary_conditions(
    grid: grids.Grid,
    inlet_velocity: Tuple[float, float] = (1.0, 0.0),
    square_centers: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (
        0.4,
        0.5,
    ),
    square_halfwidths: Union[float, Sequence[float]] = 0.05,
    immersed_bc_values: float = 0.0,
    periodic_y: bool = False,
) -> velocity_bc_types:
    """Create separate velocity boundary conditions for u and v components in 2d von Karman vortex street simulation with multiple square obstacles.

    Args:
        grid: Computational grid
        inlet_velocity: Inlet velocity (u, v) components
        square_centers: Center coordinates of square obstacles. Can be:
            - A single tuple (x, y) for one obstacle
            - A sequence of tuples [(x1, y1), (x2, y2), ...] for multiple obstacles
        square_halfwidths: Half-width of square obstacles. Can be:
            - A single float for all obstacles (if multiple centers provided)
            - A sequence of floats [w1, w2, ...] matching the number of centers
        immersed_bc_values: Value enforced inside solid regions
        periodic_y: If True, use periodic boundary conditions for upper and lower walls (y-direction)
    """
    if grid.ndim != 2:
        raise ValueError(
            "Karman vortex boundary conditions are only valid for 2D grids"
        )

    # Normalize inputs: convert single values to sequences
    # Check if it's a single center tuple (x, y) or a sequence of centers
    if isinstance(square_centers, tuple):
        if len(square_centers) == 2 and isinstance(square_centers[0], (int, float)):
            # Single center provided as tuple (x, y)
            centers = [square_centers]
        else:
            # Tuple of tuples (multiple centers)
            centers = list(square_centers)
    elif isinstance(square_centers, (list, Sequence)):
        # Sequence (list) of centers
        centers = list(square_centers)  # type: ignore[arg-type]
    else:
        raise TypeError(
            f"square_centers must be a tuple (x, y) or a sequence of tuples, got {type(square_centers)}"
        )

    num_obstacles = len(centers)

    # Normalize halfwidths
    if isinstance(square_halfwidths, (int, float)):
        # Single value provided, use for all obstacles
        halfwidths = [float(square_halfwidths)] * num_obstacles
    else:
        # Sequence provided
        halfwidths = list(square_halfwidths)
        if len(halfwidths) != num_obstacles:
            raise ValueError(
                f"Number of halfwidths ({len(halfwidths)}) must match number of centers ({num_obstacles})"
            )

    # Set y-direction boundary types based on periodic_y
    if periodic_y:
        u_y_type = (BCType.PERIODIC, BCType.PERIODIC)
        v_y_type = (BCType.PERIODIC, BCType.PERIODIC)
        u_y_values = (None, None)  # Periodic BCs have None values
        v_y_values = (None, None)
    else:
        u_y_type = (
            BCType.NEUMANN,
            BCType.NEUMANN,
        )  # y-direction: slip walls (zero normal gradient)
        v_y_type = (
            BCType.DIRICHLET,
            BCType.DIRICHLET,
        )  # y-direction: no penetration at walls
        u_y_values = (0.0, 0.0)  # type: ignore[assignment]
        v_y_values = (0.0, 0.0)  # type: ignore[assignment]

    # U-velocity boundary conditions
    u_types = (
        (
            BCType.DIRICHLET,
            BCType.NEUMANN,
        ),  # x-direction: inlet Dirichlet, outlet Neumann
        u_y_type,
    )
    u_values = (
        (inlet_velocity[0], 0.0),  # x-direction: inlet u-velocity, zero gradient outlet
        u_y_values,
    )

    # V-velocity boundary conditions
    v_types = (
        (
            BCType.DIRICHLET,
            BCType.NEUMANN,
        ),  # x-direction: inlet Dirichlet, outlet Neumann
        v_y_type,
    )
    v_values = (
        (inlet_velocity[1], 0.0),  # x-direction: inlet v-velocity, zero gradient outlet
        v_y_values,
    )

    u_bc = ImmersedBoundaryConditions(
        types=u_types,
        values=u_values,
        center=tuple(centers),
        radius=tuple(halfwidths),
        num_obstacles=num_obstacles,
        shape="square",
        immersed_bc_value=immersed_bc_values,
        grid=grid,
        offset=(1, 0.5),
    )

    v_bc = ImmersedBoundaryConditions(
        types=v_types,
        values=v_values,
        center=tuple(centers),
        radius=tuple(halfwidths),
        num_obstacles=num_obstacles,
        shape="square",
        immersed_bc_value=immersed_bc_values,
        grid=grid,
        offset=(0.5, 1),
    )

    return u_bc, v_bc


def karman_vortex_multiple_squares_boundary_conditions(
    grid: grids.Grid,
    inlet_velocity: Tuple[float, float] = (1.0, 0.0),
    inlet_pressure: float = 0.0,
    outlet_pressure: float = 0.0,
    square_centers: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (
        0.4,
        0.5,
    ),
    square_halfwidths: Union[float, Sequence[float]] = 0.05,
    immersed_bc_value: float = 0.0,
    periodic_y: bool = False,
) -> Tuple[velocity_bc_types, pressure_bc_types]:
    """Create complete set of boundary conditions for Kármán vortex street with multiple square obstacles.

    Args:
        grid: Computational grid
        inlet_velocity: Inlet velocity (u, v) components
        inlet_pressure: Pressure at inlet
        outlet_pressure: Pressure at outlet
        square_centers: Center coordinates of square obstacles. Can be:
            - A single tuple (x, y) for one obstacle
            - A sequence of tuples [(x1, y1), (x2, y2), ...] for multiple obstacles
        square_halfwidths: Half-width of square obstacles. Can be:
            - A single float for all obstacles (if multiple centers provided)
            - A sequence of floats [w1, w2, ...] matching the number of centers
        immersed_bc_value: Value enforced inside solid regions
        periodic_y: If True, use periodic boundary conditions for upper and lower walls (y-direction)

    Returns:
        Tuple of (velocity_bc, pressure_bc) boundary conditions
    """
    u_bc, v_bc = karman_vortex_multiple_squares_velocity_boundary_conditions(
        grid,
        inlet_velocity,
        square_centers,
        square_halfwidths,
        immersed_bc_value,
        periodic_y=periodic_y,
    )

    # Set pressure y-direction BC based on periodic_y
    if periodic_y:
        p_y_type = (BCType.PERIODIC, BCType.PERIODIC)
        p_y_values = (None, None)
    else:
        p_y_type = (BCType.NEUMANN, BCType.NEUMANN)
        p_y_values = (0.0, 0.0)  # type: ignore[assignment]

    p_bc = ConstantBoundaryConditions(
        types=(
            (
                BCType.NEUMANN,
                BCType.DIRICHLET,
            ),
            p_y_type,
        ),
        values=((inlet_pressure, outlet_pressure), p_y_values),
    )
    return (u_bc, v_bc), p_bc
