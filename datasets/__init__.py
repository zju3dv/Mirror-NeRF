from .blender import BlenderDataset
from .real_colmap import RealDatasetColmap
from .real_arkit import RealDatasetARKit

dataset_dict = {
    "blender": BlenderDataset,
    "real_colmap": RealDatasetColmap,
    "real_arkit": RealDatasetARKit,
}
