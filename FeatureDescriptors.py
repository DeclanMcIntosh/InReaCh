import abc

import numpy as np
import torch
import tqdm

import copy
import os
import pickle
from typing import List, Union

import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F

class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)

class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.
        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.
        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)

class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.
        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.
        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            #for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
            for _ in range(num_coreset_samples):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)

class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1).cuda()
  
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x

class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)

class Feautre_Descriptor(abc.ABC):
    def __init__(self,
                 model : torch.nn.Module,
                 image_size: tuple = (224,224,3), 
                 flatten_output: bool = True,
                 positional_embeddings: float = 5.0,
                 pretrain_embed_dimension: int = 1024,
                 target_embed_dimension: int = 1024,
                 agg_stride: int = 1,
                 agg_size: int = 3):
        self.model = model
        self.flatten_output = flatten_output
        self.positional_embeddings = positional_embeddings
        self.pretrain_embed_dimension = pretrain_embed_dimension
        self.target_embed_dimension = target_embed_dimension
        
        # Determine Model Output Sizes
        test_image = torch.from_numpy(np.transpose(np.zeros(shape=(1,*image_size)), axes=[0,3,1,2])).float().cuda()
        #print(test_image.size())
        features = self.model(test_image)
        self.feature_size = [(features[layer].size()[2],features[layer].size()[3]) for layer in features.keys()]
        self.feature_dimensions = [features[layer].size()[1] for layer in features.keys()]
        self.patch_shapes = [x[1] for x in features]

        self.patch_maker = PatchMaker(patchsize=agg_size, stride=agg_stride)
        self.agg_preprocessing = Preprocessing([features[layer].size()[1] for layer in features.keys()], self.pretrain_embed_dimension)
        self.pre_adapt_aggregator = Aggregator(target_dim=self.target_embed_dimension)

    def generate_descriptors(self, images:np.ndarray, quite: bool = False ):
        with torch.no_grad():
            output = []
            for _, image in enumerate(tqdm.tqdm(images, ncols=100, desc = 'Gen Feature Descriptors', disable=quite)):  
                features = self.model(self.image_net_norm(image).cuda())
                features = [features[layer] for layer in features.keys()]
                features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
                patch_shapes = [x[1] for x in features]
                features = [x[0] for x in features]
                ref_num_patches = patch_shapes[0]
                for i in range(1, len(features)):
                    _features = features[i]
                    patch_dims = patch_shapes[i]

                    _features = _features.reshape(
                        _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                    )
                    _features = _features.permute(0, -3, -2, -1, 1, 2)
                    perm_base_shape = _features.shape
                    _features = _features.reshape(-1, *_features.shape[-2:])
                    _features = F.interpolate(
                        _features.unsqueeze(1),
                        size=(ref_num_patches[0], ref_num_patches[1]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    _features = _features.squeeze(1)
                    _features = _features.reshape(
                        *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                    )
                    _features = _features.permute(0, -2, -1, 1, 2, 3)
                    _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                    features[i] = _features
                features = [x.reshape(-1, *x.shape[-3:]) for x in features]

                features = self.agg_preprocessing(features)
                features = self.pre_adapt_aggregator(features)
                features = torch.reshape(features, (*self.feature_size[0],self.target_embed_dimension))
                output.append(features.unsqueeze(0).cpu()) # Store on CPU to preserve GPU Memory
                del features
                del _features
                torch.cuda.empty_cache() 

            output = torch.cat(output,axis=0)
            output = torch.permute(output, (0,3,1,2))
            shape = output.size()
            if self.positional_embeddings > 0:
                with torch.no_grad():
                    positions = torch.arange(0,shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(2)
                    positions = torch.mul(positions,self.positional_embeddings/shape[2])
                    positions = positions.repeat(shape[0],1,shape[3],1)
                    positions = torch.cat([positions,torch.transpose(positions,2,3)],axis=1) 
                    output = torch.cat([output,positions],axis=1)

            if self.flatten_output:
                shape = output.size()
                output = torch.reshape(output, (shape[0],shape[1],shape[2]*shape[3]))

        return output

    def image_net_norm(self, image: np.ndarray):
        image = image/255.
        image = (image - [0.456, 0.406, 0.485])/[0.229, 0.224, 0.225] 
        image = np.expand_dims(image, axis=0)
        image = np.transpose(image,axes=[0,3,1,2])
        if len(image.shape) == 3:
            image = np.expand_dims(image,axis=0)
        return torch.tensor(image).float()
