# Glare-Spot-Detection-and-Face-Parsing-Implementation

This implementation includes a mask creation program and an inpainting program that inpaints the target mask! The idea is that the glare spots identified on human nose always give the facial 3D Reconstruction a hard time. Removing the bright spots on the nose can possibly make the reconstruction process better. So, the overview of the process is:

1. Create a glare spot mask given an inputted image
2. Use the outputted mask to run throguh the inpainting program
3. The bright spots should then be inpainted and "removed".

- Reproducer: Dennis Chao

### Installation
Requires `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -e segment_anything
python -m pip install -r lama/requirements.txt 
```
In Windows, we recommend you to first install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and 
open `Anaconda Powershell Prompt (miniconda3)` as administrator.
Then pip install [./lama_requirements_windows.txt](lama_requirements_windows.txt) instead of 
[./lama/requirements.txt](lama%2Frequirements.txt).

### Mask Creation Usage
To create a target mask, it requires an input image and bright_spot_detect.py to output a target mask. The input image must be in the same directory as the bright_spot_detect.py. Just remember to change the input path inside the program to the name of the image you want to use.

### Remove Glare Spot Usage
Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) and [LaMa](./lama/README.md) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [big-lama](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg)), and put them into `./pretrained_models`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`.

For MobileSAM, the sam_model_type should use "vit_t", and the sam_ckpt should use "./weights/mobile_sam.pt".
For the MobileSAM project, please refer to [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
```
bash script/remove_anything.sh

```
Specify an image and a mask, and Remove Bright Spot will remove the glare point at the mask specified. The input image and the mask must be inside the example folder as shown in the command. The outputted image will then be saved in the results folder. So, once you have created a bright spot mask using bright_spot_detect.py, you can then move the mask to the example folder along with the original input image. The remove_anything program will then function as the command.
```bash
python remove_anything.py --input_img .\example\input1.jpg --mask_img .\example\mask1.jpg --out_dir .\results --lama_config .\lama\configs\prediction\default.yaml --lama_ckpt .\pretrained_models\big-lama
```