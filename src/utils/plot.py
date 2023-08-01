import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional

import gc
import numpy as np
import pandas as pd
import PIL
import pytorch3d.io
import rerun as rr
import seaborn as sns
import torch
import trimesh
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt
from torch.nn import functional as F

from .logger import print_log

VIZ_HEIGHT = 300
VIZ_WIDTH = 500
VIZ_MAX_IMG_SIZE = 128
VIZ_POINTS = 5000

# TODO refactor code into Visualizer base class, RerunVisualizer and VisdomVisualizer


class RerunVisualizer:
    def __init__(self, rrd_filename: Optional[str], run_dir: str) -> None:
        self.run_dir = run_dir
        self.glb_files_dir = os.path.join(self.run_dir, "glb_files")

        rr.init("Differential Block World", spawn=rrd_filename is None)

        if rrd_filename is not None:
            rr.save(os.path.join(run_dir, rrd_filename))

    def upload_images(self, images, title, ncol=None, max_size=VIZ_MAX_IMG_SIZE):
        # TODO log images to rerun
        pass

    def log_p3d_mesh(self, entity_path, p3d_mesh):
        file_name = entity_path.replace("/", "-") + ".glb"
        glb_path = os.path.join(self.glb_files_dir, file_name)
        num_verts = p3d_mesh.num_verts_per_mesh()[0]
        tm_mesh = trimesh.Trimesh(
            vertices=p3d_mesh.verts_packed().numpy(force=True),
            faces=p3d_mesh.faces_packed().numpy(force=True),
        )
        tm_mesh.visual = trimesh.visual.TextureVisuals(
            uv=p3d_mesh.textures.verts_uvs_padded()[0, :num_verts].numpy(force=True),
            image=PIL.Image.fromarray(
                np.uint8(p3d_mesh.textures.maps_padded()[0].numpy(force=True) * 255)
            ),
        )
        tm_mesh.vertex_normals  # this computes vertex normals
        tm_mesh.export(glb_path)
        rr.log_mesh_file(entity_path, rr.MeshFormat.GLB, mesh_path=glb_path)

    def log_model(self, cur_iter, model):
        """Log current meshes."""
        rr.set_time_sequence("iteration", cur_iter)
        os.makedirs(self.glb_files_dir, exist_ok=True)
        blocks = model.build_blocks(
            world_coord=True, filter_transparent=False, as_scene=False
        )
        for i, block in enumerate(blocks):
            self.log_p3d_mesh(f"world/blocks/#{i}", block)

        ground = model.build_ground(world_coord=True)
        self.log_p3d_mesh("world/ground", ground)

        background = model.build_bkg(world_coord=True)
        self.log_p3d_mesh("world/background", background)

        # TODO separately log with thresholded opacity

    def log_dataset(self, cur_iter, dataset):
        rr.set_time_sequence("iteration", cur_iter)
        for image_id, (image_dict, label) in enumerate(dataset):
            _, height, width = image_dict["imgs"].shape
            scale = min(height, width) / 2.0
            fx = image_dict["K"][0, 0] * scale
            fy = image_dict["K"][1, 1] * scale
            cx = -image_dict["K"][0, 2] * scale + width / 2.0
            cy = -image_dict["K"][1, 2] * scale + height / 2.0
            translation = image_dict["T"].clone()
            rotation = image_dict["R"].clone()
            # row vector convention (pytorch3d) -> column vector convention (rerun)
            rotation = rotation.T
            rr.log_pinhole(
                f"world/train_images/#{image_id}",
                width=width,
                height=height,
                focal_length_px=(fx, fy),
                principal_point_px=(cx, cy),
                camera_xyz="LUF",  # see https://pytorch3d.org/docs/cameras
            )
            rr.log_transform3d(
                f"world/train_images/#{image_id}",
                rr.TranslationAndMat3(translation=translation, matrix=rotation),
                from_parent=True,  # pytorch3d uses camera from world
            )
            rr.log_image(
                f"world/train_images/#{image_id}", image_dict["imgs"].permute(1, 2, 0)
            )

    def log_scalars(self, cur_iter, named_values, title, *_, **__):
        rr.set_time_sequence("iteration", cur_iter)
        for name, value in named_values:
            rr.log_scalar(title + "/" + name, value)

    def upload_barplot(self, named_values, title):
        # TODO log barplot to rerun
        pass

    def upload_pointcloud(self, points, colors=None, title=None):
        # TODO log pointcloud to rerun
        pass


class VisdomVisualizer:
    def __init__(self, port, run_dir):
        if port is not None:
            import visdom

            os.environ["http_proxy"] = ""  # XXX set to solve proxy issues
            visualizer = visdom.Visdom(
                port=port, env=f"{run_dir.parent.name}_{run_dir.name}"
            )
            visualizer.delete_env(visualizer.env)  # Clean env before plotting
            print_log(f"Visualizer initialised at {port}")
        else:
            visualizer = None
            print_log("No visualizer initialized")
        self.visualizer = visualizer

    def upload_images(self, images, title, ncol=None, max_size=VIZ_MAX_IMG_SIZE):
        if self.visualizer is None:
            return None
        if max(images.shape[2:]) > max_size:
            images = F.interpolate(
                images, size=max_size, mode="bilinear", align_corners=False
            )
        ncol = ncol or len(images)
        self.visualizer.images(
            images.clamp(0, 1),
            win=title,
            nrow=ncol,
            opts={
                "title": title,
                "store_history": True,
                "width": VIZ_WIDTH,
                "height": VIZ_HEIGHT,
            },
        )

    def log_scalars(self, cur_iter, named_scalars, title, colors=None):
        if self.visualizer is None:
            return None
        names, values = map(list, zip(*named_scalars))
        y, x = [values], [[cur_iter] * len(values)]
        self.visualizer.line(
            y,
            x,
            win=title,
            update="append",
            opts={
                "title": title,
                "linecolor": colors,
                "legend": names,
                "width": VIZ_WIDTH,
                "height": VIZ_HEIGHT,
            },
        )

    def upload_barplot(self, named_values, title):
        if self.visualizer is None:
            return None
        names, values = map(list, zip(*named_values))
        self.visualizer.bar(
            values,
            win=title,
            opts={"title": title, "width": VIZ_HEIGHT, "height": VIZ_HEIGHT},
        )

    def upload_pointcloud(self, points, colors=None, title=None):
        if self.visualizer is None:
            return None
        if len(points) > VIZ_POINTS:
            indices = torch.randperm(len(points))[:VIZ_POINTS]
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
        self.visualizer.scatter(
            points,
            colors,
            win=title,
            opts={
                "title": title,
                "width": VIZ_HEIGHT,
                "height": VIZ_HEIGHT,
                "markersize": 2,
            },
        )


@contextmanager
def seaborn_context(n_colors=10):
    with sns.color_palette("colorblind", n_colors), sns.axes_style(
        "white", {"axes.grid": True, "legend.frameon": True}
    ):
        yield


def get_fancy_cmap():
    colors = sns.color_palette("hls", 21)
    gold = mplcolors.to_rgb("gold")
    colors = [gold] + colors[3:] + colors[:2]
    raw_cmap = mplcolors.LinearSegmentedColormap.from_list("Custom", colors)

    def cmap(values):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return raw_cmap(values)[:, :3]

    return cmap


def plot_lines(
    df,
    columns,
    title,
    figsize=(10, 5.625),
    drop_na=True,
    colors=None,
    style=None,
    unit_yaxis=False,
    lw=2,
):
    if not isinstance(columns, (list, tuple)):
        columns = [columns]
    if colors is None:
        colors = [None] * len(columns)

    with seaborn_context(len(columns)):
        fig, ax = plt.subplots(figsize=figsize)
        for y, color in zip(columns, colors):
            kwargs = {"ax": ax, "linewidth": lw, "style": style}
            if color is not None:
                kwargs["color"] = color
            if drop_na:
                s = df[y].dropna()
                if len(s) > 0:
                    s.plot(**kwargs)
            else:
                df[y].plot(**kwargs)
        if unit_yaxis:
            ax.set_ylim((0, 1))
            ax.set_yticks([k / 10 for k in range(11)])
        ax.grid(axis="x", which="both", color="0.5", linestyle="--", linewidth=0.5)
        ax.grid(axis="y", which="major", color="0.5", linestyle="-", linewidth=0.5)
        ax.legend(framealpha=1, edgecolor="0.3", fancybox=False)
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()

    return fig


def plot_bar(df, title, figsize=(10, 5.625), unit_yaxis=False):
    assert isinstance(df, pd.Series)
    with seaborn_context(1):
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind="bar", ax=ax, edgecolor="k", width=0.8, rot=0, linewidth=1)
        if unit_yaxis:
            ax.set_ylim((0, 1))
            ax.set_yticks([k / 10 for k in range(11)])
        ax.grid(axis="x", which="both", color="0.5", linestyle="--", linewidth=0.5)
        ax.grid(axis="y", which="major", color="0.5", linestyle="-", linewidth=0.5)
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()

    return fig


def plot_img_grid(images, nrow=None, ncol=None, scale=4, wspace=0.05, hspace=0.05):
    if isinstance(images, (list, tuple)):
        assert len(images) > 0
        if isinstance(images[0], PIL.Image.Image):
            images = list(map(np.asarray, images))
        elif isinstance(images[0], torch.Tensor):
            images = [i.cpu().numpy() for i in images]
        if images[0].shape[-1] > 3:  # XXX images are CHW
            images = [i.transpose(1, 2, 0) for i in images]

    elif len(images.shape) == 4 and images.shape[-1] > 3:  # images are BCHW
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().permute(0, 2, 3, 1).clamp(0, 1)
        else:
            images = images.transpose(0, 2, 3, 1).clip(0, 1)

    if ncol is None:
        if nrow is None:
            nrow, ncol = 1, len(images)
        else:
            ncol = (len(images) - 1) // nrow + 1
    else:
        nrow = (len(images) - 1) // ncol + 1
    gridspec_kw = {"wspace": wspace, "hspace": hspace}
    fig, axarr = plt.subplots(
        nrow, ncol, figsize=(ncol * scale, nrow * scale), gridspec_kw=gridspec_kw
    )
    for ax, img in zip(axarr.ravel(), images):
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_axis_off()


def create_visualizer(cfg, run_dir):
    kwargs = deepcopy(cfg["training"]["visualizer"] or {})
    name = kwargs.pop("name")
    visualizer = get_visualizer(name)(**kwargs, run_dir=run_dir)
    print_log(f'Visualizer "{name}" init: kwargs={cfg["training"]["visualizer"]}')
    return visualizer


def get_visualizer(name):
    if name is None:
        name = "rerun"
    return {
        "visdom": VisdomVisualizer,
        "rerun": RerunVisualizer,
    }[name]
