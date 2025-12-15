import gc
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import wandb
import os
from monai.utils import set_determinism
import numpy as np
import segmentation_models_pytorch as sm
from monai.networks import nets
import yaml
from PIL import Image

def plot_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.title('Predicted Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()

def getNetworkArch(arch_name, kargs, size, backbone, encoder_weights):
    input_channel_names = ['n_input_channels', 'in_channels' ]
    output_class_names = ['num_classes']
    if arch_name not in ['UNet', 'AttentionUnet']:
       kargs.pop('strides' )
       kargs.pop('channels')

    if arch_name in ['UNETR', 'SwinUNETR']:
       kargs['spatial_dims'] = 2
       kargs['img_size'] = (size,size)
    
    if arch_name == 'FlexibleUNet':
       kargs['backbone'] = backbone
    # network= None

    try:
      # logging.info("Monai Network architcture ")
      network= getattr(nets, arch_name)(**kargs)
      # nets.AttentionUnet(spatial_dims=2, in_channels=1, out_channels=5,
      #                       channels=(64, 128, 256, 512,1024), strides=(2, 2, 2,2))
    except:
      # logging.info("SM Network architcture ")
      try:
        network= getattr(sm, arch_name)(in_channels=kargs['in_channels'], classes= kargs['out_channels'],
                                        encoder_name=backbone, encoder_weights=encoder_weights )
      except:
        # logging.info("SM Deeplab ")
        network= getattr(sm, arch_name)(in_channels=kargs['in_channels'], classes= kargs['out_channels'],
                                         encoder_name=backbone, encoder_weights=None )
    return network

###################################################################
############### Loading the model from the wandb###################
def loadingModel(net, run, device, MODALITY_TYPE = "T1"):
    model_name = f'xai_lss/Segmentation_{MODALITY_TYPE}/{net}'
    network_name = model_name.split('/')[-1].replace(":", "_")
    artifact = run.use_artifact(model_name, type='model') # SwinUNETR:v9 # UnetPlusPlus:v0  SwinUNETR:v12
    artifact_dir =artifact.download()
    # print(artifact_dir)
    artifact_dir_content = os.listdir(artifact_dir)
    # print()
    if 'config.yaml' in artifact_dir_content:
        config_file = artifact_dir_content.pop(artifact_dir_content.index('config.yaml'))
    model_name = artifact_dir_content[0]
    model_path = os.path.join(artifact_dir,model_name)
    # print(model_path)
    producer_run = artifact.logged_by()
    wandb.config.update(producer_run.config, allow_val_change=True)
    config = wandb.config

    set_determinism(config.seed,use_deterministic_algorithms=False)

    archt_cofiguration = dict(spatial_dims=2, in_channels=config.input_channels , out_channels=config.out_channels,
                            channels=tuple(config.channels), strides=tuple([config.strides]* (len(config.channels)-1)))

    model= getNetworkArch(config.arch, archt_cofiguration, config.size, config.backbone,
                            encoder_weights=config.encoder_weights).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
###################################################################
################### Loading the model locally #####################
def InitModel(model_name, viewe_plane, device):
#   device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  all_models_folder = f"./artifacts_2/{viewe_plane}"
  model_folder = os.path.join(all_models_folder, model_name)
  model_folder_content = os.listdir(model_folder)
  config_file = model_folder_content.pop(model_folder_content.index('config.yaml'))
  config_file_path = os.path.join(model_folder, config_file)
  model_path = os.path.join(model_folder, model_folder_content[0])
  # print(model_path)

  bases_model_config = loadConfigFile(config_file_path)
  # set_determinism(config.seed,use_deterministic_algorithms=False)
  # print()
  # print(bases_model_config)
  # print()

  archt_cofiguration = dict(spatial_dims=2, in_channels=bases_model_config['input_channels'] , out_channels=bases_model_config['out_channels'],
                          channels=tuple(bases_model_config['channels']), strides=tuple([bases_model_config['strides']]* (len(bases_model_config['channels'])-1)))

  model= getNetworkArch(bases_model_config['arch'], archt_cofiguration, bases_model_config['size'], bases_model_config['backbone'],
                          encoder_weights=bases_model_config['encoder_weights']).to(device)
  # print(model_path)
  # print(model)

  model.load_state_dict(torch.load(model_path, map_location=device))

  return model, bases_model_config

def loadConfigFile(yaml_file_path):
    with open(yaml_file_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
    config_dict = {key: value['value'] for (key, value) in data.items() if 'wandb' not in key}
    return config_dict


def preProcessImage(image_byte):
    pil_image = Image.open(image_byte)
    pil_image= pil_image.resize(size=(256, 256), resample= Image.BICUBIC)

    image = np.array(pil_image)
    print("Image shape", image.shape)
    if (image > 1).any():
        image = image / 255.0
        image.astype(np.float32)

    if image.ndim == 2:
        image_tensor= torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    else:
    #    image_tensor= torch.from_numpy(image)
    #    print("image_tensor shape", image_tensor.shape)
       image_tensor= torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    image_tensor = image_tensor.float().contiguous()
    return image_tensor

@torch.inference_mode()
def forwaredPass(models_list, viewe_plane, image,  amp=True):
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_preds = []
    ################ loading the model ################
    loaded_models = {}
    for  model_name in models_list:
        model,  bases_model_config = InitModel(model_name, viewe_plane, device)
        model.eval()
        loaded_models[model_name] = model

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for model_name, model in loaded_models.items():
            with torch.no_grad():
                print(f"Model {model_name} is running")
                print("Image shape", image.shape)
                mask_pred = model(image)
                mask_pred = F.sigmoid(mask_pred)
                mask_pred = torch.where(mask_pred > 0.5, mask_pred, torch.tensor(0.0, device=mask_pred.device)).squeeze(dim=1)
                mask_preds.append(mask_pred)

        mask_pred_combined = torch.stack(mask_preds, dim=1)
        concatenated_feature_map_prob = F.softmax(mask_pred_combined, dim=2)
        average_feature_map= torch.mean(concatenated_feature_map_prob, dim=1)
        mask_pred = average_feature_map.argmax(dim=1).squeeze().cpu().numpy()
        # plot_mask(mask_pred)

    for model in loaded_models.values():
        model.cpu()
        del model
    torch.cuda.empty_cache()
    gc.collect()

    return mask_pred

def ensembleInference(image, VIEWE_PLANE):
    top_models_dict = {
       "axial": ['SwinUNETR_v13', 'UnetPlusPlus_v1', 'AttentionUnet_v0'],
       "sagittal": ['UnetPlusPlus_v0', 'Unet_v2', 'SwinUNETR_v0'],
    #    "sagittal": ['UnetPlusPlus_v0', 'Unet_v2'],

    }
    models_list =top_models_dict[VIEWE_PLANE] ## ALERT depending on the modalities best models

    pred_mask = forwaredPass(models_list, VIEWE_PLANE, image)

    return pred_mask

