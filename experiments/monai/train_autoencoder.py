import functools
import glob
import scripts
import torch

from monai import transforms
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.utils import set_determinism
from monai.handlers import CheckpointSaver, StatsHandler, TensorBoardStatsHandler, from_engine

from generative.losses import PerceptualLoss
from monai.networks.nets.autoencoderkl import AutoencoderKL
from monai.networks.nets.patchgan_discriminator import PatchDiscriminator


print("start")
bundle_root = "."
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt_dir = bundle_root + '/models/autoencoder'
tf_dir = bundle_root + '/eval'
dataset_dir = "data/brats-2021-msd"
pretrained = False
perceptual_loss_model_weights_path = ""
train_batch_size = 2
lr = 1e-05
train_patch_size = [
    112,
    128,
    80
]
channel = 0
spacing = [
    1.1,
    1.1,
    1.1
]
spatial_dims = 3
image_channels = 1
latent_channels = 8

discriminator = PatchDiscriminator(
        spatial_dims = spatial_dims,
        num_layers_d = 3,
        channels = 32,
        in_channels = 1,
        out_channels = 1,
        norm= "INSTANCE"
    )
autoencoder = AutoencoderKL(
        spatial_dims = spatial_dims,
        in_channels = image_channels,
        out_channels = image_channels,
        latent_channels = latent_channels,
        channels= [
            64,
            128,
            256
        ],
        num_res_blocks = 2,
        norm_num_groups = 32,
        norm_eps = 1e-06,
        attention_levels= [
            False,
            False,
            False
        ],
        with_encoder_nonlocal_attn = False,
        with_decoder_nonlocal_attn = False,
        include_fc= False
        )
perceptual_loss = PerceptualLoss(
        spatial_dims = spatial_dims,
        network_type = "resnet50",
        is_fake_3d = True,
        fake_3d_ratio = 0.2,
        pretrained = pretrained,
        pretrained_path = perceptual_loss_model_weights_path,
        pretrained_state_dict_key = "state_dict"
    )
dnetwork = discriminator.to(device)
gnetwork = autoencoder.to(device)
loss_perceptual = perceptual_loss.to(device)
doptimizer = torch.optim.Adam(
        params = dnetwork.parameters(),
        lr = lr
    )
goptimizer = torch.optim.Adam(
        params = gnetwork.parameters(),
        lr = lr
    )
preprocessing_transforms = [
            transforms.LoadImaged(
            keys = ["image"]
            ),
            transforms.EnsureChannelFirstd(
            keys = ["image"]
),
            transforms.Lambdad(
            keys = ["image"],
            func = lambda x: x[channel, :, :, :]
            ),
            transforms.EnsureChannelFirstd(
            keys = ["image"],
            channel_dim = "no_channel"
            ),
            transforms.EnsureTyped(
            keys = ["image"]
            ),
            transforms.Orientationd(
            keys = ["image"],
            axcodes = "RAS"
            ),
            transforms.Spacingd(
            keys = ["image"],
            pixdim = spacing,
            mode = "bilinear"
            )
    ]
final_transforms= [
            transforms.ScaleIntensityRangePercentilesd(
            keys = ["image"],
            lower = 0,
            upper = 99.5,
            b_min = 0,
            b_max = 1
            )
    ]
    # train
crop_transforms= [
                transforms.RandSpatialCropd(
                keys = "image",
                roi_size = train_patch_size,
                random_size = False
                )
        ]
preprocessing= transforms.Compose(
            transforms = preprocessing_transforms + crop_transforms + final_transforms
        )
print("before dataset")
dataset= DecathlonDataset(
            root_dir = dataset_dir,
            task = "Task01_BrainTumour",
            section = "training",
            cache_rate = 0.0,
            num_workers = 0,
            download = False,
            transform = preprocessing
        )
print("after dataset")
dataloader=DataLoader(
            dataset = dataset,
            batch_size = train_batch_size,
            shuffle = True,
            num_workers = 0
        )
print("after dataloader")
handlers= [
                CheckpointSaver(
                save_dir = ckpt_dir,
                save_dict= {
                    "model": gnetwork
                },
                save_interval= 0,
                save_final = True,
                epoch_level = True,
                final_filename = "model_autoencoder2.pt"
            ),
            StatsHandler(
                tag_name = "train_loss",
                output_transform = lambda x: from_engine(['g_loss'], first=True)(x)[0]
            ),
            TensorBoardStatsHandler(
                log_dir = tf_dir,
                tag_name = "train_loss",
                output_transform = lambda x: from_engine(['g_loss'], first=True)(x)[0]
            )
        ]
trainer = scripts.ldm_trainer.VaeGanTrainer(
            device = device,
            max_epochs = 1500,
            train_data_loader = dataloader,
            g_network = gnetwork,
            g_optimizer = goptimizer,
            g_loss_function = functools.partial(scripts.losses.generator_loss, disc_net=dnetwork, loss_perceptual=loss_perceptual),
            d_network = dnetwork,
            d_optimizer = doptimizer,
            d_loss_function =functools.partial(scripts.losses.discriminator_loss, disc_net=dnetwork),
            d_train_steps = 5,
            g_update_latents = True,
            latent_shape = latent_channels,
            key_train_metric = None,
            train_handlers = handlers
        )

set_determinism(seed=0)
trainer.run()