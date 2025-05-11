# Author: 
# Tiago Assis
# Faculty of Sciences, University of Lisbon
# March 2025

import argparse
import numpy as np
import nrrd
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes


def main(brain_mask_path: str, tumor_mask_path: str) -> None:
    vector_generator = SurgicalVectorGenerator(brain_mask_path=brain_mask_path, tumor_mask_path=tumor_mask_path)
    vector_generator.run()


class SurgicalVectorGenerator:
    """
    Class to generate gravity vectors representing possible surgical entry directions
    based on brain and tumor mask volumes.
    """

    def __init__(self, brain_mask_path: str, tumor_mask_path: str, angle_variability: float = 45.0, 
                 num_samples: int = 5, seed: None | int = None) -> None:
        """
        Initializes the SurgicalVectorGenerator class with the paths to the brain and tumor masks,
        the angle variability, the number of samples, and the random seed.

        Args:
            brain_mask_path (str): Path to the brain segmentation file (.nrrd).
            tumor_mask_path (str): Path to the tumor segmentation file (.nrrd).
            angle_variability (float, optional): Maximum allowed angle deviation, in degrees, used to generate new random vectors from the main vector. Defaults to 45.0.
            num_samples (int, optional): Number of new gravity vectors to generate. Defaults to 5.
            seed (None | int, optional): Random seed for reproducibility, dictating the sampled angles for the new vectors generated. Defaults to None.
        """
        self.brain_mask_path = brain_mask_path
        self.tumor_mask_path = tumor_mask_path
        self.angle_variability = angle_variability
        self.num_samples = num_samples
        self.seed = seed

        self.brain_mask, self.t_header = nrrd.read(brain_mask_path)
        self.tumor_mask, self.b_header = nrrd.read(tumor_mask_path)

    def compute_centroid(self, mask: np.ndarray, header: dict) -> np.ndarray:
        """
        Computes the centroid of the non-zero region in the tumor segmentations, transformed to physical space.

        Args:
            mask (np.ndarray): 3D binary segmentation of the tumor.
            header (dict): Header from the NRRD file containing spatial transformation info.

        Returns:
            np.ndarray: 3D coordinates of the tumor centroid in physical space.
        """
        # Find the array indices where the tumor is present
        coords = np.argwhere(mask > 0)
        # Find the tumor center coordinates by calculating the mean index of each axis
        centroid = np.mean(coords, axis=0)
        # Make sure the coordinates are always in physical space, so the later code makes sense for every case
        reoriented_centroid = centroid @ header["space directions"] + header["space origin"]
        return reoriented_centroid

    def extract_brain_surface(self, brain_mask: np.ndarray, header: dict) -> np.ndarray:
        """
        Extracts the 3D coordinates of the brain surface using the marching cubes algorithm.

        Args:
            brain_mask (np.ndarray): 3D binary mask of the brain.
            header (dict): Header from the NRRD file containing spatial transformation info.

        Returns:
            np.ndarray: Nx3 array of vertices representing the brain surface in physical space.
        """
        # Find the boundary of the brain (brain surface) using a marching cubes algorithm
        verts, _, _, _ = marching_cubes(brain_mask)
        verts = verts @ header["space directions"] + header["space origin"]
        return verts

    def find_closest_surface_point(self, tumor_center: np.ndarray, brain_surface: np.ndarray) -> np.ndarray:
        """
        Finds the closest point on the brain surface to the tumor center, ensuring it's either above or at the same level.

        Args:
            tumor_center (np.ndarray): 3D coordinates of the tumor center.
            brain_surface (np.ndarray): Nx3 array of brain surface vertices.

        Returns:
            np.ndarray: 3D coordinates of the closest valid surface point.
        """
        # Calculate the distance between every surface point and the tumor center
        distances = cdist([tumor_center], brain_surface)
        # Sort the surface points by their distance
        closest_idxs = np.argsort(distances)
        # Keep looking for the next closest surface point, if the Z-axis of the current closest point is below the tumor center 
        # (prevents the entry point being under the brain, which is physically not possible).
        # Quick constraint here because in generate_gravity_directions() there's an additional one.
        for i in range(closest_idxs.shape[1]):
            closest_surface_point = brain_surface[closest_idxs[0][i]]
            if closest_surface_point[2] > tumor_center[2]:
                break
        return closest_surface_point

    def compute_main_vector(self, tumor_center: np.ndarray, closest_surface_point: np.ndarray) -> np.ndarray:
        """
        Computes the normalized vector from the tumor center to the brain surface.

        Args:
            tumor_center (np.ndarray): 3D coordinates of the tumor center.
            closest_surface_point (np.ndarray): 3D coordinates of a brain surface point.

        Returns:
            np.ndarray: Normalized vector from tumor to surface.
        """
        # Compute the vector has the difference between the closest surface point coordinates and the tumor center coordinates
        vector = closest_surface_point - tumor_center
        norm_vector = vector / np.linalg.norm(vector)
        # As the vector goes from the tumor center to the surface, invert it to define the gravity direction 
        #gravity_coeffs = -norm_vector
        return norm_vector

    def generate_gravity_directions(self, base_vector: np.ndarray, angle_variability: float=45.0, 
                                    num_samples: int=5, seed: None | int=None) -> np.ndarray:
        """
        Generates additional gravity vectors around the base direction, constrained by angle variability, minimum separation, and biophysical accuracy.

        Args:
            base_vector (np.ndarray): Base gravity direction vector (normalized).
            angle_variability (float, optional): Maximum allowed angle deviation, in degrees, used to generate new random vectors from the main vector. Defaults to 45.0.
            num_samples (int, optional): Number of new gravity vectors to generate. Defaults to 5.
            seed (None | int, optional): Random seed for reproducibility, dictating the sampled angles for the new vectors generated. Defaults to None.

        Returns:
            np.ndarray: Array of shape (num_samples, 3) with varied gravity vectors.
        """
        # For reproducible testing purposes
        rng = np.random.default_rng(seed)

        # Convert coordinates of the main gravity vector to spherical coordinates
        theta = np.arctan2(base_vector[1], base_vector[0])
        phi = np.arcsin(base_vector[2])

        # Cnvert degrees to radians
        angle_var_rad = np.radians(angle_variability)

        # The constraint for all the angles to be separated by a certain minimum angle
        # sometimes led to infinite loops if the angle variability is small for the number of samples.
        # (i.e., any angle randomly sampled would not fulfill the constraint, so an infinite loop would occur)
        # This equation was manually tested to fit to points that allowed the code to run properly.
        min_angle_separation = 150 * num_samples ** -1.6 ##

        min_angle = np.radians(min_angle_separation)

        theta_variations = [theta]
        phi_variations = [phi]

        # Sample random angles but only append those that are at least min_angle radians apart
        while len(theta_variations) < num_samples:
            theta_variation = rng.uniform(-angle_var_rad, angle_var_rad)
            if all(abs(theta_variation - a) >= min_angle for a in theta_variations):
                theta_variations.append(theta_variation)

        while len(phi_variations) < num_samples:
            phi_variation = rng.uniform(-angle_var_rad, angle_var_rad)
            if all(abs(phi_variation - a) >= min_angle for a in phi_variations):
                phi_variations.append(phi_variation)
        
        # Turn radians to coefficients,
        # which are normalized because the base vector is normalized
        gravity_directions = [-base_vector]
        for i in range(1,num_samples):
            new_theta = theta + theta_variations[i]
            new_phi = phi + phi_variations[i]
            # get x,y,z directions
            new_x = np.cos(new_phi) * np.cos(new_theta)
            new_y = np.cos(new_phi) * np.sin(new_theta)
            new_z = np.sin(new_phi)
            gravity_coeffs = np.array([-new_x, -new_y, -new_z])
            # Clip Z-axis to ensure that the new entry vectors don't come from under the brain.
            # It allows a bit of movement to occur in the Z-axis (in comparison to the first constraint), but clips it if it is too big (> 0.2).
            # 0.2 is a value that seemed okay to use as a constraint, based on my testing. Can be changed.
            if gravity_coeffs[2] > 0.2:
                gravity_coeffs[2] = 0.2
            gravity_directions.append(gravity_coeffs)

        return np.array(gravity_directions)

    def run(self) -> None:
        """
        Generates and saves gravity vectors based on the given input files and parameters.
        """
        print("Finding hypothetical surgical entry points and corresponding gravity vectors...\n")
        tumor_center = self.compute_centroid(self.tumor_mask, self.t_header)
        brain_surface = self.extract_brain_surface(self.brain_mask, self.b_header)
        closest_surface_point = self.find_closest_surface_point(tumor_center, brain_surface)
        norm_vector = self.compute_main_vector(tumor_center, closest_surface_point)
        gravity_directions = self.generate_gravity_directions(norm_vector, self.angle_variability, self.num_samples, self.seed)

        np.savetxt("./mtled_outputs/gravity_vectors.txt", gravity_directions, fmt = "%.3f %.3f %.3f", newline='\n')
        print("\tGravity vectors saved.")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    It performs validation and ensures the constraints are met. The following arguments are parsed:
    - `brain_mask_path` (str): Path to the brain segmentation .nrrd file (positional argument).
    - `tumor_mask_path` (str): Path to the tumor segmentation .nrrd file (positional argument).
    - `angle_variability` (float, optional): The angle variability for sampling (default: 45.0).
    - `num_samples` (int, optional): Number of samples to generate (default: 5).
    - `seed` (int or None, optional): The random seed for reproducibility (default: None).

    Returns:
        argparse.Namespace: The parsed command-line arguments as an object.
    
    Raises:
        AssertionError: If any of the following conditions are violated:
            - `brain_mask_path` or `tumor_mask_path` does not end with `.nrrd`.
            - `angle_variability` is not a float or int, or it's outside the range (-180, 180).
            - `num_samples` is not a positive integer.
            - `seed` is neither `None` nor an integer.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('brain_mask_path', type=str, help="Path to the brain segmentation .nrrd file.")
    parser.add_argument('tumor_mask_path', type=str, help="Path to the tumor segmentation .nrrd file.")
    parser.add_argument('--angle_variability', type=float, default=45.0, help="Angle variability (float or int). Default is 45.0 degrees.")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of samples to generate. Default is 5.")
    parser.add_argument('--seed', type=int, default=None, nargs='?', help="Random seed for sampling. Default is None.")

    args = parser.parse_args()

    assert args.brain_mask_path.lower().endswith(".nrrd"), "The brain segmentation file must be in .nrrd format."
    assert args.tumor_mask_path.lower().endswith(".nrrd"), "The tumor segmentation file must be in .nrrd format."
    assert isinstance(args.angle_variability, (float, int)), "The angle variability must be either an integer or a float."
    assert -180 < args.angle_variability < 180, "The angle variability cannot exceed 180 degrees in both directions."
    assert isinstance(args.num_samples, int), "The number of samples must be an integer."
    assert args.num_samples > 0, "The number of samples must be an integer greater than 0."
    assert isinstance(args.seed, (None, int)), "The random seed must be either None or an integer."

    return args


if __name__ == '__main__':
    args = parse_arguments()
    vector_generator = SurgicalVectorGenerator(
        brain_mask_path=args.brain_mask_path,
        tumor_mask_path=args.tumor_mask_path,
        angle_variability=args.angle_variability,
        num_samples=args.num_samples,
        seed=args.seed
    )
    vector_generator.run()
