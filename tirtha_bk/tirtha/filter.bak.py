"""
3D Gaussian converter to densify the gaussians and
to remove the extreneous floaters from the scene.

"""

import sys
import signal
import numpy as np
from .utils import Logger
from collections import deque
from multiprocessing import Pool, cpu_count
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors


class Filter:
    def __init__(self, input_path, output_path, runDir, log_path):
        self.Input_Path = input_path
        self.Output_Path = output_path
        self.source_format = "3dgs"
        self.Target_formet = "3dgs"
        self.Density_filter = [1, 0.3]
        self.Remove_flyers = [25, 1.0]
        self.Debug = False
        self.RGB = False
        self.BBOX = None
        self.runDir = runDir
        self.log_path = log_path
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.logger = Filter.logger = Logger(
            log_path=self.log_path,
            name=f"Filtering",
        )

        # if not self.Output_Path.lower().endswith('.ply'):
        #     self.Output_Path += '.ply'

        data = PlyData.read(self.Input_Path)

        if isinstance(data, PlyData) and "vertex" in data:
            self.logger.info(
                f"Number of vertices in the header: {len(data['vertex'].data)}"
            )
            structured_data = data["vertex"].data

        else:
            self.logger.info("Error: Data format is not PlyData with a 'vertex' field.")
            return

        try:
            with Pool(initializer=init_worker) as pool:
                # If the bbox argument is provided, extract its values
                # bbox_values = self.BBOX if self.BBOX else None

                # For PlyData, access the vertex data
                self.data = data["vertex"].data
                # baseconverter = BaseConverter(data_to_convert)
                # Call the convert function and pass the data to convert
                converted_instance = self.Convert(
                    self.source_format,
                    self.Target_formet,
                    process_rgb=self.RGB,
                    density_filter=self.Density_filter,
                    remove_flyers=self.Remove_flyers,
                    bbox=self.BBOX,
                    pool=pool,
                )
                # converted_data = converted_instance.converted_data
                converted_data = converted_instance

        except KeyboardInterrupt:
            self.logger.info("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
            pool.join()
            sys.exit(-1)

        # Check if the conversion actually happened and save the result
        if isinstance(converted_data, np.ndarray):
            # Save the converted data to the output file
            PlyData(
                [PlyElement.describe(converted_data, "vertex")], byte_order="="
            ).write(self.Output_Path)
            self.logger.info(f"Conversion completed and saved to {self.Output_Path}.")
        else:
            self.logger.info("Conversion was skipped.")

    def Convert(self, source_format, target_format, **kwargs):

        self.logger.info(
            f"[DEBUG] Starting conversion from {source_format} to {target_format}..."
        )

        # Apply optional pre-processing steps using process_data (newly added)
        self.Process_data(
            bbox=kwargs.get("bbox"),
            apply_density_filter=kwargs.get("density_filter"),
            remove_flyers=kwargs.get("remove_flyers"),
        )

        # Conversion operations
        process_rgb_flag = kwargs.get("process_rgb", False)

        self.logger.info("[DEBUG] Applying operations on 3DGS data...")
        # if not any(kwargs.values()):  # If no flags are provided
        #     self.logger.info("[INFO] No flags provided. The conversion will not happen as the output would be identical to the input.")
        #     self.converted_data = data['vertex'].data
        # else:
        #     self.converted_data = converter.to_3dgs()

        self.logger.info("[DEBUG] Starting conversion from 3DGS to 3DGS...")

        # Load vertices from the updated data after all filters
        # vertices = self.data
        vertices = self.data
        self.logger.info(f"[DEBUG] Loaded {len(vertices)} vertices.")

        # Create a new structured numpy array for 3DGS format
        dtype_3dgs = self.define_dtype(
            has_scal=False, has_rgb=False
        )  # Define 3DGS dtype without any prefix
        converted_data = np.zeros(vertices.shape, dtype=dtype_3dgs)

        # Use the helper function to copy the data from vertices to converted_data
        self.copy_data_with_prefix_check(
            vertices, converted_data, ["", "scal_", "scalar_", "scalar_scal_"]
        )

        self.logger.info("[DEBUG] Data copying completed.")
        self.logger.info("[DEBUG] Sample of converted data (first 5 rows):")
        # if DEBUG:
        for i in range(5):
            print(converted_data[i])

        self.logger.info("[DEBUG] Conversion from 3DGS to 3DGS completed.")
        return converted_data

    def Process_data(self, bbox=None, apply_density_filter=None, remove_flyers=None):

        if bbox:
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
            self.crop_by_bbox(min_x, min_y, min_z, max_x, max_y, max_z)
            self.logger.info("[DEBUG] Bounding box cropped.")

        # Apply density filter if parameters are provided
        # if apply_density_filter:
        # Unpack parameters, applying default values if not all parameters are given
        voxel_size, threshold_percentage = (apply_density_filter + [1.0, 0.32])[
            :2
        ]  # Defaults to 1.0 and 0.32 if not provided
        self.apply_density_filter(
            voxel_size=float(voxel_size),
            threshold_percentage=float(threshold_percentage),
        )
        self.logger.info("[DEBUG] Density filter applied.")

        # Remove flyers if parameters are provided
        # if remove_flyers:
        # Example: expecting remove_flyers to be a list or tuple like [k, threshold_factor]
        # Provide default values if necessary
        k, threshold_factor = (remove_flyers + [25, 1.0])[
            :2
        ]  # Defaults to 25 and 1.0 if not provided
        self.remove_flyers(k=int(k), threshold_factor=float(threshold_factor))
        self.logger.info("[DEBUG] Flyers removed.")

    def apply_density_filter(self, voxel_size=1.0, threshold_percentage=0.32):
        self.logger.info("[DEBUG] Executing 'apply_density_filter' function...")
        # Ensure self.data is a numpy structured array
        if not isinstance(self.data, np.ndarray):
            raise TypeError("self.data must be a numpy structured array.")

        vertices = (
            self.data
        )  # This assumes self.data is already a numpy structured array

        # Convert threshold_percentage into a ratio
        threshold_ratio = threshold_percentage / 100.0

        # Parallelized voxel counting
        voxel_counts = self.parallel_voxel_counting(vertices, voxel_size)

        threshold = int(len(vertices) * threshold_ratio)
        dense_voxels = {k: v for k, v in voxel_counts.items() if v >= threshold}

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

        # Convert the filtered vertices list to a numpy structured array
        self.data = np.array(filtered_vertices, dtype=vertices.dtype)

        # Informative print statement
        self.logger.info(
            f"After density filter, retained {len(self.data)} out of {len(vertices)} vertices."
        )

        # Since we're working with numpy arrays, just return self.data
        return self.data

    def crop_by_bbox(self, min_x, min_y, min_z, max_x, max_y, max_z):
        # Perform cropping based on the bounding box
        self.data = self.data[
            (self.data["x"] >= min_x)
            & (self.data["x"] <= max_x)
            & (self.data["y"] >= min_y)
            & (self.data["y"] <= max_y)
            & (self.data["z"] >= min_z)
            & (self.data["z"] <= max_z)
        ]
        # Print the number of vertices after cropping
        self.logger.info(f"[DEBUG] Number of vertices after cropping: {len(self.data)}")

        # Informative print statement
        self.logger.info(f"After cropping, retained {len(self.data)} vertices.")

        return self.data

    def remove_flyers(self, k=25, threshold_factor=10.5, chunk_size=50000):
        self.logger.info("[DEBUG] Executing 'remove_flyers' function...")

        # Ensure self.data is a numpy structured array
        if not isinstance(self.data, np.ndarray):
            raise TypeError("self.data must be a numpy structured array.")

        # Extract vertex data from the current object's data
        vertices = self.data
        num_vertices = len(vertices)

        # Display the number of input vertices
        self.logger.info(f"[DEBUG] Number of input vertices: {num_vertices}")

        # Adjust k based on the number of vertices
        k = max(
            3, min(k, num_vertices // 100)
        )  # Example: ensure k is between 3 and 1% of the total vertices
        self.logger.info(f"[DEBUG] Adjusted k to: {k}")

        # Number of chunks
        num_chunks = (num_vertices + chunk_size - 1) // chunk_size  # Ceiling division
        masks = []

        # Create a pool of workers
        num_cores = max(1, cpu_count() - 1)  # Leave one core free
        with Pool(processes=num_cores, initializer=init_worker) as pool:
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

                # Compute K-Nearest Neighbors for the chunk
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

        # Apply the mask to the vertices and store the result in self.data
        self.data = vertices[combined_mask]

        self.logger.info(
            f"After removing flyers, retained {np.count_nonzero(combined_mask)} out of {num_vertices} vertices."
        )
        return self.data

    @staticmethod
    def define_dtype(has_scal, has_rgb=False):
        print("[DEBUG] Executing 'define_dtype' function...")

        prefix = "scalar_scal_" if has_scal else ""
        print(f"[DEBUG] Prefix determined as: {prefix}")

        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            (f"{prefix}f_dc_0", "f4"),
            (f"{prefix}f_dc_1", "f4"),
            (f"{prefix}f_dc_2", "f4"),
            *[(f"{prefix}f_rest_{i}", "f4") for i in range(45)],
            (f"{prefix}opacity", "f4"),
            (f"{prefix}scale_0", "f4"),
            (f"{prefix}scale_1", "f4"),
            (f"{prefix}scale_2", "f4"),
            (f"{prefix}rot_0", "f4"),
            (f"{prefix}rot_1", "f4"),
            (f"{prefix}rot_2", "f4"),
            (f"{prefix}rot_3", "f4"),
        ]
        print("[DEBUG] Main dtype constructed.")

        if has_rgb:
            dtype.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
            print("[DEBUG] RGB fields added to dtype.")

        print("[DEBUG] 'define_dtype' function completed.")
        return dtype, prefix

    @staticmethod
    def copy_data_with_prefix_check(source, target, possible_prefixes):
        print("[DEBUG] Executing 'copy_data_with_prefix_check' function...")

        """
        Given two structured numpy arrays (source and target), copy the data from source to target.
        If a field exists in source but not in target, this function will attempt to find the field
        in target by adding any of the possible prefixes to the field name.
        """
        for name in source.dtype.names:
            if name in target.dtype.names:
                target[name] = source[name]
            else:
                copied = False
                for prefix in possible_prefixes:
                    # If the field starts with the prefix, try the field name without the prefix
                    if name.startswith(prefix):
                        stripped_name = name[len(prefix) :]
                        if stripped_name in target.dtype.names:
                            target[stripped_name] = source[name]
                            copied = True
                            break
                    # If the field doesn't start with any prefix, try adding the prefix
                    else:
                        prefixed_name = prefix + name
                        if prefixed_name in target.dtype.names:
                            print(
                                f"[DEBUG] Copying data from '{name}' to '{prefixed_name}'"
                            )
                            target[prefixed_name] = source[name]
                            copied = True
                            break
                ##if not copied:
                ##    print(f"Warning: Field {name} not found in target.")

    @staticmethod
    def knn_worker(args):
        print(f"[DEBUG] Executing 'knn_worker' function for vertex: {args[0]}...")

        """Utility function for parallel KNN computation."""
        coords, tree, k = args
        coords = coords.reshape(1, -1)  # Reshape to a 2D array
        distances, _ = tree.kneighbors(coords)
        avg_distance = np.mean(distances[:, 1:])

        print(
            f"[DEBUG] Average distance computed for vertex: {args[0]} is {avg_distance}."
        )
        return avg_distance

    def parallel_voxel_counting(self, vertices, voxel_size=1.0):
        self.logger.info("[DEBUG] Executing 'parallel_voxel_counting' function...")

        """Counts the number of points in each voxel in a parallelized manner."""
        num_processes = cpu_count()
        chunk_size = len(vertices) // num_processes
        chunks = [
            vertices[i : i + chunk_size] for i in range(0, len(vertices), chunk_size)
        ]

        num_cores = max(1, cpu_count() - 1)
        with Pool(processes=num_cores, initializer=init_worker) as pool:
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

        self.logger.info(
            f"[DEBUG] Voxel counting completed with {len(total_voxel_counts)} unique voxels found."
        )
        return total_voxel_counts

    @staticmethod
    def get_neighbors(voxel_coords):
        print(f"[DEBUG] Getting neighbors for voxel: {voxel_coords}...")

        """Get the face-touching neighbors of the given voxel coordinates."""
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
    def count_voxels_chunk(vertices_chunk, voxel_size):
        print("[DEBUG] Executing 'count_voxels_chunk' function for a chunk...")

        """Count the number of points in each voxel for a chunk of vertices."""
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

        print(f"[DEBUG] Chunk processed with {len(voxel_counts)} voxels counted.")
        return voxel_counts


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# def debug_print(message):
#     self.logger.info(message)

# if __name__ == "__main__":

#     Filter(input_path, output_path, log_path)
