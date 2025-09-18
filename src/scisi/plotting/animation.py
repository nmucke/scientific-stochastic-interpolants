"""
Animation utilities for creating MP4 videos from torch tensors.
"""

import os
from typing import Any, List, Optional, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Colormap


def tensor_to_mp4(
    tensor: torch.Tensor,
    file_name: str,
    fps: int = 30,
    colormap: Union[str, Colormap] = "viridis",
    dpi: int = 100,
    bitrate: int = 1800,
    codec: str = "h264",
    normalize: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Create an MP4 video from a torch tensor.

    Args:
        tensor: Input tensor of shape [height, width, time_steps]
        file_name: Output filename (should include .mp4 extension)
        fps: Frames per second for the output video
        colormap: Matplotlib colormap name or Colormap object
        dpi: Dots per inch for the video frames
        bitrate: Video bitrate in kbps
        codec: Video codec to use
        normalize: Whether to normalize the data to [0, 1] range
        vmin: Minimum value for colormap scaling (if None, uses data min)
        vmax: Maximum value for colormap scaling (if None, uses data max)

    Raises:
        ValueError: If tensor doesn't have the expected shape
        RuntimeError: If video creation fails
    """
    if tensor.dim() != 3:
        raise ValueError(
            f"Expected 3D tensor [height, width, time_steps], got {tensor.dim()}D tensor with shape {tensor.shape}"
        )

    height, width, time_steps = tensor.shape

    # Convert to numpy and ensure contiguous memory
    data = tensor.detach().cpu().numpy()

    # Normalize data if requested
    if normalize:
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:  # Avoid division by zero
            data = (data - data_min) / (data_max - data_min)

    # Set up colormap scaling
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    # Create figure and axis
    # fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    fig, ax = plt.subplots()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")  # Remove axes for cleaner video

    # Initialize image
    im = ax.imshow(data[:, :, 0], cmap=colormap, vmin=vmin, vmax=vmax, origin="lower")

    def animate(frame: int) -> List[plt.AxesImage]:
        """Update the image for each frame."""
        im.set_array(data[:, :, frame])
        return [im]

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=time_steps, interval=1000 / fps, blit=True, repeat=True
    )

    # Save as MP4
    try:
        # Ensure output directory exists
        os.makedirs(
            os.path.dirname(file_name) if os.path.dirname(file_name) else ".",
            exist_ok=True,
        )

        # Save the animation
        writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate, codec=codec)
        anim.save(file_name, writer=writer)

        print(f"Video saved successfully: {file_name}")

    except Exception as e:
        raise RuntimeError(f"Failed to create video: {str(e)}")

    finally:
        plt.close(fig)


def tensors_to_mp4(
    tensors: List[torch.Tensor],
    file_name: str,
    fps: int = 30,
    colormaps: Union[str, Colormap, List[Union[str, Colormap]]] = "viridis",
    titles: Optional[List[str]] = None,
    dpi: int = 100,
    bitrate: int = 1800,
    codec: str = "h264",
    normalize: Union[bool, List[bool]] = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[tuple] = None,
    show_colorbar: bool = True,
    colorbar_label: str = "Value",
) -> None:
    """
    Create an MP4 video from multiple torch tensors displayed side by side.

    Args:
        tensors: List of input tensors, each of shape [height, width, time_steps]
        file_name: Output filename (should include .mp4 extension)
        fps: Frames per second for the output video
        colormaps: Single colormap or list of colormaps for each tensor
        titles: Optional list of titles for each subplot
        dpi: Dots per inch for the video frames
        bitrate: Video bitrate in kbps
        codec: Video codec to use
        normalize: Whether to normalize each tensor to [0, 1] range (single bool or list)
        vmin: Minimum value for colormap scaling (applied to all tensors, from first tensor if None)
        vmax: Maximum value for colormap scaling (applied to all tensors, from first tensor if None)
        figsize: Figure size as (width, height) tuple. If None, auto-calculated.
        show_colorbar: Whether to display a colorbar on the left side
        colorbar_label: Label for the colorbar

    Raises:
        ValueError: If tensors don't have consistent dimensions or invalid parameters
        RuntimeError: If video creation fails
    """
    if not tensors:
        raise ValueError("No tensors provided")

    num_tensors = len(tensors)

    # Validate all tensors have the same dimensions
    first_shape = tensors[0].shape
    if len(first_shape) != 3:
        raise ValueError(
            f"All tensors must be 3D [height, width, time_steps], got {len(first_shape)}D tensor"
        )

    height, width, time_steps = first_shape

    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.shape != first_shape:
            raise ValueError(
                f"Tensor {i} has shape {tensor.shape}, expected {first_shape}"
            )

    # Convert colormaps to list
    if isinstance(colormaps, (str, Colormap)):
        colormaps = [colormaps] * num_tensors
    elif len(colormaps) != num_tensors:
        raise ValueError(
            f"Number of colormaps ({len(colormaps)}) must match number of tensors ({num_tensors})"
        )

    # Convert normalize to list
    if isinstance(normalize, bool):
        normalize = [normalize] * num_tensors
    elif len(normalize) != num_tensors:
        raise ValueError(
            f"Number of normalize values ({len(normalize)}) must match number of tensors ({num_tensors})"
        )

    # Validate vmin/vmax (now single values applied to all tensors)
    if vmin is not None and not isinstance(vmin, (int, float)):
        raise ValueError("vmin must be a single float value or None")

    if vmax is not None and not isinstance(vmax, (int, float)):
        raise ValueError("vmax must be a single float value or None")

    # Set up titles
    if titles is None:
        titles = [f"Tensor {i+1}" for i in range(num_tensors)]
    elif len(titles) != num_tensors:
        raise ValueError(
            f"Number of titles ({len(titles)}) must match number of tensors ({num_tensors})"
        )

    # Convert tensors to numpy and normalize
    data_list = []
    global_vmin = None
    global_vmax = None

    for i, (tensor, norm) in enumerate(zip(tensors, normalize)):
        data = tensor.detach().cpu().numpy()

        if norm:
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)

        # For the first tensor, determine global vmin/vmax
        if i == 0:
            if vmin is None:
                global_vmin = data.min()
            else:
                global_vmin = vmin

            if vmax is None:
                global_vmax = data.max()
            else:
                global_vmax = vmax

        data_list.append(data)

    # Use the same vmin/vmax for all tensors (from first tensor)
    vmin_all = [global_vmin] * num_tensors
    vmax_all = [global_vmax] * num_tensors

    # Calculate figure size if not provided
    if figsize is None:
        aspect_ratio = (width * num_tensors) / height
        fig_width = min(20, max(8, aspect_ratio * 4))
        fig_height = fig_width / aspect_ratio
        figsize = (fig_width, fig_height)

    # Create figure and subplots with space for colorbar if needed
    if show_colorbar:
        fig_width_with_colorbar = figsize[0] + 1.0  # Add 1 inch for colorbar
        fig, axes = plt.subplots(
            1, num_tensors, figsize=(fig_width_with_colorbar, figsize[1]), dpi=dpi
        )
        # Adjust subplot layout to make room for colorbar
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)
    else:
        fig, axes = plt.subplots(1, num_tensors, figsize=figsize, dpi=dpi)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    if num_tensors == 1:
        axes = [axes]

    # Initialize images
    images = []
    for i, (ax, data, colormap) in enumerate(zip(axes, data_list, colormaps)):
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_title(titles[i])
        ax.axis("off")

        im = ax.imshow(
            data[:, :, 0],
            cmap=colormap,
            vmin=vmin_all[i],
            vmax=vmax_all[i],
            origin="lower",
        )
        images.append(im)

    # Add a single colorbar on the left side using the first image (if requested)
    if show_colorbar:
        cbar_ax = fig.add_axes([0.02, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        colorbar = fig.colorbar(images[0], cax=cbar_ax, orientation="vertical")
        colorbar.set_label(colorbar_label, rotation=270, labelpad=20)

    def animate(frame: int) -> List[plt.AxesImage]:
        """Update all images for each frame."""
        for im, data in zip(images, data_list):
            im.set_array(data[:, :, frame])
        return images

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=time_steps, interval=1000 / fps, blit=True, repeat=True
    )

    # Save as MP4
    try:
        # Ensure output directory exists
        os.makedirs(
            os.path.dirname(file_name) if os.path.dirname(file_name) else ".",
            exist_ok=True,
        )

        # Save the animation
        writer = animation.FFMpegWriter(fps=fps, bitrate=bitrate, codec=codec)
        anim.save(file_name, writer=writer)

        print(f"Multi-tensor video saved successfully: {file_name}")

    except Exception as e:
        raise RuntimeError(f"Failed to create video: {str(e)}")

    finally:
        plt.close(fig)


def create_animation_from_tensor(
    tensor: torch.Tensor,
    file_name: str,
    fps: int = 30,
    colormap: Union[str, Colormap] = "viridis",
    **kwargs: Any,
) -> None:
    """
    Alias for tensor_to_mp4 for backward compatibility.

    Args:
        tensor: Input tensor of shape [height, width, time_steps]
        file_name: Output filename (should include .mp4 extension)
        fps: Frames per second for the output video
        colormap: Matplotlib colormap name or Colormap object
        **kwargs: Additional arguments passed to tensor_to_mp4
    """
    tensor_to_mp4(tensor, file_name, fps, colormap, **kwargs)


def create_animation_from_tensors(
    tensors: List[torch.Tensor],
    file_name: str,
    fps: int = 30,
    colormaps: Union[str, Colormap, List[Union[str, Colormap]]] = "viridis",
    show_colorbar: bool = True,
    colorbar_label: str = "Value",
    **kwargs: Any,
) -> None:
    """
    Alias for tensors_to_mp4 for convenience.

    Args:
        tensors: List of input tensors, each of shape [height, width, time_steps]
        file_name: Output filename (should include .mp4 extension)
        fps: Frames per second for the output video
        colormaps: Single colormap or list of colormaps for each tensor
        show_colorbar: Whether to display a colorbar on the left side
        colorbar_label: Label for the colorbar
        **kwargs: Additional arguments passed to tensors_to_mp4
    """
    tensors_to_mp4(
        tensors,
        file_name,
        fps,
        colormaps,
        show_colorbar=show_colorbar,
        colorbar_label=colorbar_label,
        **kwargs,
    )
