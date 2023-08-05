import humanize
import math
import torch
import torch.distributed as dist
import warnings
from itertools import islice
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange


class Queue(Dataset):
    """Adapted from TorchIO Queue: \
        https://github.com/fepegar/torchio/blob/main/torchio/data/queue.py"""

    def __init__(
        self,
        subjects_dataset: Dataset,
        max_length: int,
        samples_per_volume: int,
        sampler: object,
        num_workers: int = 0,
        shuffle_subjects: bool = True,
        shuffle_patches: bool = True,
        start_background: bool = True,
        dist_train: bool = True,
        verbose: bool = False,
    ):
        self.subjects_dataset = subjects_dataset
        self.max_length = max_length
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_patches = shuffle_patches
        self.samples_per_volume = samples_per_volume
        self.sampler = sampler
        self.num_workers = num_workers
        self.dist_train = dist_train
        self.verbose = verbose
        self._subjects_iterable = None
        if self.dist_train:
            self.dist_sampler = DistributedSampler(self.subjects_dataset)
        if start_background:
            self._initialize_subjects_iterable()
        self.patches_list = []
        self.num_sampled_patches = 0
        self.epoch = 0

    def __len__(self):
        return self.iterations_per_epoch

    def __getitem__(self, _):
        # There are probably more elegant ways of doing this
        if not self.patches_list:
            self._print("Patches list is empty.")
            self._fill()
        sample_patch = self.patches_list.pop(0)
        self.num_sampled_patches += 1
        return sample_patch

    def __repr__(self):
        attributes = [
            f"max_length={self.max_length}",
            f"num_subjects={self.num_subjects}",
            f"num_patches={self.num_patches}",
            f"samples_per_volume={self.samples_per_volume}",
            f"num_sampled_patches={self.num_sampled_patches}",
            f"iterations_per_epoch={self.iterations_per_epoch}",
        ]
        attributes_string = ", ".join(attributes)
        return f"Queue({attributes_string})"

    def _print(self, *args):
        if self.verbose:
            print(*args)

    def _initialize_subjects_iterable(self):
        self._subjects_iterable = self._get_subjects_iterable()

    @property
    def subjects_iterable(self):
        if self._subjects_iterable is None:
            self._initialize_subjects_iterable()
        return self._subjects_iterable

    @property
    def num_subjects(self) -> int:
        return len(self.subjects_dataset)

    @property
    def num_patches(self) -> int:
        return len(self.patches_list)

    @property
    def iterations_per_epoch(self) -> int:
        if self.dist_train:
            return math.ceil(self.num_subjects / dist.get_world_size()) * self.samples_per_volume
        else:
            return self.num_subjects * self.samples_per_volume

    def _fill(self) -> None:
        if self.max_length % self.samples_per_volume != 0:
            message = (
                f"Queue length ({self.max_length})"
                " not divisible by the number of"
                f" patches per volume ({self.samples_per_volume})"
            )
            warnings.warn(message, RuntimeWarning)

        # If there are e.g. 4 subjects and 1 sample per volume and max_length
        # is 6, we just need to load 4 subjects, not 6
        max_num_subjects_for_queue = self.max_length // self.samples_per_volume
        num_subjects_for_queue = min(self.num_subjects, max_num_subjects_for_queue)

        self._print(f"Filling queue from {num_subjects_for_queue} subjects...")
        if self.verbose:
            iterable = trange(num_subjects_for_queue, leave=False)
        else:
            iterable = range(num_subjects_for_queue)
        for _ in iterable:
            subject = self._get_next_subject()
            if self.sampler is not None:
                iterable = self.sampler(subject)
                patches = list(islice(iterable, self.samples_per_volume))
            else:
                patches = [subject]
            self.patches_list.extend(patches)
        if self.shuffle_patches:
            self._shuffle_patches_list()

    def _shuffle_patches_list(self):
        indices = torch.randperm(self.num_patches)
        self.patches_list = [self.patches_list[i] for i in indices]

    def _get_next_subject(self):
        # A StopIteration exception is expected when the queue is empty
        try:
            subject = next(self.subjects_iterable)
        except StopIteration as exception:
            self._print("Queue is empty:", exception)
            if self.dist_train:
                self.epoch += 1
                self.dist_sampler.set_epoch(self.epoch)
            self._initialize_subjects_iterable()
            subject = next(self.subjects_iterable)
        except AssertionError as exception:
            if "can only test a child process" in str(exception):
                message = (
                    "The number of workers for the data loader used to pop"
                    " patches from the queue should be 0. Is it?"
                )
                raise RuntimeError(message) from exception
        return subject

    @staticmethod
    def _get_first_item(batch):
        return batch[0]

    def _get_subjects_iterable(self):
        # I need a DataLoader to handle parallelism
        # But this loader is always expected to yield single subject samples
        self._print(f"\nCreating subjects loader with {self.num_workers} workers")
        if self.dist_train:
            subjects_loader = DataLoader(
                self.subjects_dataset,
                num_workers=self.num_workers,
                batch_size=1,
                collate_fn=self._get_first_item,
                sampler=self.dist_sampler,
            )
        else:
            subjects_loader = DataLoader(
                self.subjects_dataset,
                num_workers=self.num_workers,
                batch_size=1,
                collate_fn=self._get_first_item,
                shuffle=self.shuffle_subjects,
            )
        return iter(subjects_loader)

    def get_max_memory(self, subject) -> int:
        """Get the maximum RAM occupied by the patches queue in bytes.
        Args:
            subject: Sample subject to compute the size of a patch.
        """
        images_channels = 0
        if subject is None:
            subject = self.subjects_dataset[0]
        for image in subject.get_images(intensity_only=False):
            images_channels += len(image.data)
        voxels_in_patch = int(self.sampler.patch_size.prod() * images_channels)
        bytes_per_patch = 4 * voxels_in_patch  # assume float32
        return int(bytes_per_patch * self.max_length)

    def get_max_memory_pretty(self, subject) -> str:
        """Get human-readable maximum RAM occupied by the patches queue.
        Args:
            subject: Sample subject to compute the size of a patch.
        """
        memory = self.get_max_memory(subject=subject)
        return humanize.naturalsize(memory, binary=True)
