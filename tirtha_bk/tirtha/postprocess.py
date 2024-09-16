"""
3D Gaussian converter to densify the gaussians and
to remove the extreneous floaters from the scene.

NOTE: Code mostly adapted from https://github.com/francescofugazzi/3dgsconverter
Licensed under the MIT License

NOTE: `run_convert()` adapted from https://github.com/antimatter15/splat/blob/main/convert.py
Licensed under the MIT License

"""

# 3DGS format
# NOTE: Kept for reference, but unused
# dtype_3dgs = [
#     ("x", "f4"),
#     ("y", "f4"),
#     ("z", "f4"),
#     ("nx", "f4"),
#     ("ny", "f4"),
#     ("nz", "f4"),
#     ("f_dc_0", "f4"),
#     ("f_dc_1", "f4"),
#     ("f_dc_2", "f4"),
#     *[("f_rest_{i}", "f4") for i in range(45)],
#     ("opacity", "f4"),
#     ("scale_0", "f4"),
#     ("scale_1", "f4"),
#     ("scale_2", "f4"),
#     ("rot_0", "f4"),
#     ("rot_1", "f4"),
#     ("rot_2", "f4"),
#     ("rot_3", "f4"),
# ]

import signal
import numpy as np
from collections import deque
from multiprocessing import Pool, cpu_count
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors

# For .splat conversion
from io import BytesIO

# Local imports
from .utils import Logger


def _init_worker():
    """
    Ignore SIGINT signals in worker processes.

    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class PostProcess:
    """
    Applies the following operations to the Gaussian Splat:
    1. Density filter
    2. Floater removal
    3. Conversion from `.ply` to `.splat` format

    """

    def __init__(self, input_path, output_path, runDir, log_path):
        self.input_path = input_path
        self.output_path = output_path
        self.dens_filt_args = [1, 0.3]  # [voxel_size, thresh_percen]
        self.rem_float_args = [25, 1.0]  # [k, threshold_factor]

        self.runDir = runDir
        self.log_path = log_path
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.logger = PostProcess.logger = Logger(
            log_path=log_path,
            name="PostProcess",
        )

        # Load the input `.ply` file
        self.data = PlyData.read(input_path)["vertex"].data
        self.logger.info(f"Number of vertices in the header: {len(self.data)}")

    def run_ops(self):
        """
        Runs the post-processing operations on the Gaussian Splat.

        """
        # Post-processing steps - updates self.data in-place
        # Apply density filter
        voxel_size, thresh_percen = self.dens_filt_args
        self.apply_density_filter(
            voxel_size=float(voxel_size),
            thresh_percen=float(thresh_percen),
        )

        # Remove floaters
        k, threshold_factor = self.rem_float_args
        self.remove_floaters(k=int(k), threshold_factor=float(threshold_factor))

        # Convert the filtered `.ply` file to `.splat` format
        self.run_convert()

    def apply_density_filter(
        self, voxel_size: float = 1.0, thresh_percen: float = 0.32
    ) -> None:
        """
        Applies a density filter to the Gaussian Splat.

        Parameters
        ----------
        voxel_size : float, optional
            The size of the voxel grid, by default 1.0
        thresh_percen : float, optional
            The threshold percentage of vertices to retain, by default 0.32

        """
        self.logger.info(f"Applying density filter with args: {self.dens_filt_args}")

        vertices = self.data
        threshold_ratio = thresh_percen / 100.0

        # Parallelized voxel counting
        voxel_counts = self.parallel_voxel_counting(vertices, voxel_size)

        threshold = int(len(vertices) * threshold_ratio)
        dense_voxels = {k: v for k, v in voxel_counts.items() if v >= threshold}

        # Find the largest cluster of dense voxels
        visited = set()
        max_cluster = set()
        for voxel in dense_voxels:
            if voxel not in visited:
                current_cluster = set()
                queue = deque([voxel])
                while queue:
                    current_voxel = queue.popleft()
                    visited.add(current_voxel)
                    current_cluster.add(current_voxel)
                    for neighbor in self.get_neighbors(current_voxel):
                        if neighbor in dense_voxels and neighbor not in visited:
                            queue.append(neighbor)
                            visited.add(neighbor)
                if len(current_cluster) > len(max_cluster):
                    max_cluster = current_cluster

        # Filter vertices to only include those in dense voxels
        filtered_vertices = [
            vertex
            for vertex in vertices
            if (
                int(vertex["x"] / voxel_size),
                int(vertex["y"] / voxel_size),
                int(vertex["z"] / voxel_size),
            )
            in max_cluster
        ]

        # Convert the filtered vertices list to a numpy structured array and update self.data
        self.data = np.array(filtered_vertices, dtype=vertices.dtype)

        self.logger.info(
            f"Density filtering retained {len(self.data)} out of {len(vertices)} vertices."
        )
        self.logger.info("Density filter applied.")

    def remove_floaters(
        self, k: int = 25, threshold_factor: float = 10.5, chunk_size: int = 50_000
    ) -> None:
        """
        Removes floaters from the Gaussian Splat.

        Parameters
        ----------
        k : int, optional
            The number of nearest neighbors to consider, by default 25
        threshold_factor : float, optional
            The factor to multiply the standard deviation of the average distances by, by default 10.5
        chunk_size : int, optional
            The size of the chunks to process in parallel, by default 50_000

        """
        self.logger.info(f"Applying floater removal with args: {self.dens_filt_args}")

        vertices = self.data
        num_vertices = len(vertices)
        self.logger.info(f"Number of input vertices: {num_vertices}")

        # Adjust k based on the number of vertices
        k = max(
            3, min(k, num_vertices // 100)
        )  # Example: ensure k is between 3 and 1% of the total vertices
        self.logger.info(
            f"Adjusted k to: {k}. Ensure `k` is between 1 to 3% of total vertices."
        )

        # Number of chunks
        num_chunks = (num_vertices + chunk_size - 1) // chunk_size
        masks = []

        num_cores = max(1, cpu_count() - 1)  # Leaves one core free
        with Pool(processes=num_cores, initializer=_init_worker) as pool:
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(
                    start_idx + chunk_size, num_vertices
                )  # Avoid going out of bounds
                chunk_coords = np.vstack(
                    (
                        vertices["x"][start_idx:end_idx],
                        vertices["y"][start_idx:end_idx],
                        vertices["z"][start_idx:end_idx],
                    )
                ).T

                # Compute k-NearestNeighbors for the chunk
                nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(
                    chunk_coords
                )
                avg_distances = pool.map(
                    self.knn_worker, [(coord, nbrs, k) for coord in chunk_coords]
                )

                # Calculate the threshold for removal based on the mean and standard deviation of the average distances
                threshold = np.mean(avg_distances) + threshold_factor * np.std(
                    avg_distances
                )

                # Create a mask for points to retain for this chunk
                mask = np.array(avg_distances) < threshold
                masks.append(mask)

        # Combine masks from all chunks
        combined_mask = np.concatenate(masks)

        # Apply the mask to the vertices and update self.data
        self.data = vertices[combined_mask]

        self.logger.info(
            f"Floater removal retained {np.count_nonzero(combined_mask)} out of {num_vertices} vertices."
        )
        self.logger.info("Floater removal applied.")

    def run_convert(self) -> None:
        """
        Converts the filtered `.ply` file to `.splat` format.

        Adapted from https://github.com/antimatter15/splat/blob/main/convert.py
        Licensed under the MIT License

        """
        self.logger.info("Converting .ply to .splat...")

        # Paths
        output_file = self.output_path

        # Cleaned data
        vertices = self.data
        sorted_indices = np.argsort(
            -np.exp(vertices["scale_0"] + vertices["scale_1"] + vertices["scale_2"])
            / (1 + np.exp(-vertices["opacity"]))
        )
        buffer = BytesIO()
        for idx in sorted_indices:
            v = vertices[idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            SH_C0 = 0.28209479177387814
            color = np.array(
                [
                    0.5 + SH_C0 * v["f_dc_0"],
                    0.5 + SH_C0 * v["f_dc_1"],
                    0.5 + SH_C0 * v["f_dc_2"],
                    1 / (1 + np.exp(-v["opacity"])),
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )

        splat_data = buffer.getvalue()

        # Save
        self.logger.info(f"Saving to {output_file}.")
        with open(output_file, "wb") as f:
            f.write(splat_data)
        self.logger.info("Converted .ply to .splat.")

    @staticmethod
    def knn_worker(args: tuple) -> float:
        """
        Utility function for parallel KNN computation.

        Parameters
        ----------
        args : tuple
            The arguments for the worker function, containing the coordinates, tree, and k (unused)

        Returns
        -------
        float
            The average distance to the k-nearest neighbors, excluding the point itself

        """
        coords, tree, _ = args
        coords = coords.reshape(1, -1)  # Reshape to a 2D array
        distances, _ = tree.kneighbors(coords)

        return np.mean(distances[:, 1:])  # Average distance excluding the point itself

    def parallel_voxel_counting(
        self, vertices: np.ndarray, voxel_size: float = 1.0
    ) -> dict:
        """
        Utility function that counts the number of points in each voxel in a parallelized manner.

        Parameters
        ----------
        vertices : np.ndarray
            The vertices to count
        voxel_size : float, optional
            The size of the voxel grid, by default 1.0

        Returns
        -------
        dict
            A dictionary of voxel coordinates to the number of points in that voxel

        """
        num_processes = cpu_count()
        chunk_size = len(vertices) // num_processes
        chunks = [
            vertices[i : i + chunk_size] for i in range(0, len(vertices), chunk_size)
        ]

        num_cores = max(1, cpu_count() - 1)
        with Pool(processes=num_cores, initializer=_init_worker) as pool:
            results = pool.starmap(
                self.count_voxels_chunk, [(chunk, voxel_size) for chunk in chunks]
            )

        # Aggregate results from all processes
        total_voxel_counts = {}
        for result in results:
            for k, v in result.items():
                if k in total_voxel_counts:
                    total_voxel_counts[k] += v
                else:
                    total_voxel_counts[k] = v

        return total_voxel_counts

    @staticmethod
    def get_neighbors(voxel_coords: tuple) -> list:
        """
        Utility function to get the face-touching neighbors of the given voxel coordinates.

        Parameters
        ----------
        voxel_coords : tuple
            The coordinates of the voxel

        Returns
        -------
        list
            A list of the face-touching neighbors of the voxel

        """
        x, y, z = voxel_coords
        neighbors = [
            (x - 1, y, z),
            (x + 1, y, z),
            (x, y - 1, z),
            (x, y + 1, z),
            (x, y, z - 1),
            (x, y, z + 1),
        ]

        return neighbors

    @staticmethod
    def count_voxels_chunk(vertices_chunk: np.ndarray, voxel_size: float) -> dict:
        """
        Utility function to count the number of points in each voxel for a chunk of vertices.

        Parameters
        ----------
        vertices_chunk : np.ndarray
            The chunk of vertices to count

        voxel_size : float
            The size of the voxel grid

        Returns
        -------
        dict
            A dictionary of voxel coordinates to the number of points in that voxel

        """
        voxel_counts = {}
        for vertex in vertices_chunk:
            voxel_coords = (
                int(vertex["x"] / voxel_size),
                int(vertex["y"] / voxel_size),
                int(vertex["z"] / voxel_size),
            )
            if voxel_coords in voxel_counts:
                voxel_counts[voxel_coords] += 1
            else:
                voxel_counts[voxel_coords] = 1

        return voxel_counts
