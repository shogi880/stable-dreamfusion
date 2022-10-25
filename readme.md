# 3D to 3D pipeline based on Stable-Dreamfusion,

A pytorch implementation of the 3D-to-3D pipeline based on [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion) text-to-3D model.

Colab notebook for usage: [![Open In Colab](https://drive.google.com/file/d/1BN9N1k8oQwXFWBF8PZElQtW3QFzGsAJG/view?usp=sharing)

# Important Notice
This project is a **work-in-progress**, welcome for any contribution and collaboration.
The main problems in stable-dreamfusion are still unsolved.
- Alleviate the multi-face Janus problem.
- Better mesh (improve the surface quality).


# Install (refer to [ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion))


```bash
git clone https://github.com/ashawkey/stable-dreamfusion.git
cd stable-dreamfusion

**Important**: To download the Stable Diffusion model checkpoint, you should provide your [access token](https://huggingface.co/settings/tokens). You could choose either of the following ways:
* Run `huggingface-cli login` and enter your token.
* Create a file called `TOKEN` under this directory (i.e., `stable-dreamfusion/TOKEN`) and copy your token into it.

### Install with pip
```bash
pip install -r requirements.txt

# install nvdiffrast for exporting textured mesh (--save_mesh)
pip install git+https://github.com/NVlabs/nvdiffrast/

```

### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
We also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
pip install ./raymarching # install to python path (you still need the raymarching/ folder, since this only installs the built extension.)
```

### Tested environments
* torch 1.12 & CUDA 11.6 on a V100.


# Usage
## Dataset
Our 3D-to-3D pipeline aims to transfer human skeletons with different poses to novel 3D assets in blender.
We collect the 100 images from 100 camera poses on a circle orbit for each pose.
The blender scripts and all dataset can be found at [google drive](https://drive.google.com/drive/folders/1lpXdhT5pFoaINw__csJhp1wtxD2DBO5n?usp=sharing).

Please download the dataset folder and place in `./stable-dreamfusion`.

## Run command

First, to pre-train a NeRF with downloaded poses dataset. (we provided the model.pth in the drive)

```bash
# pre-train NeRF
python main.py --text "pose_1" -O3 --gt_dir dataset/pose_1 --save_mesh
```

# transfer pretrained NeRF.

```bash
# --reload_model to load pretrain model from pretrain_ckpt.
python main.py --text "a raccoon astronaut" -O3 --nerf_transfer --pretrain_ckpt ./pretrain_models/model.pth --reload_model --save_mesh
```


# Acknowledgement

* The amazing pytorch-implementation project: [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion).

* The amazing original work: [_DreamFusion: Text-to-3D using 2D Diffusion_](https://dreamfusion3d.github.io/).
    ```
    @article{poole2022dreamfusion,
        author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
        title = {DreamFusion: Text-to-3D using 2D Diffusion},
        journal = {arXiv},
        year = {2022},
    }
    ```

* Huge thanks to the [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and the [diffusers](https://github.com/huggingface/diffusers) library. 

    ```
    @misc{rombach2021highresolution,
        title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
        author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
        year={2021},
        eprint={2112.10752},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

    @misc{von-platen-etal-2022-diffusers,
        author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
        title = {Diffusers: State-of-the-art diffusion models},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/huggingface/diffusers}}
    }
    ```

* The GUI is developed with [DearPyGui](https://github.com/hoffstadt/DearPyGui).
