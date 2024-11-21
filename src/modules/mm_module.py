import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import CLIPVisionModel
import torchvision.models as cv_models
from image_models import ImageEncoder
from transformers import RobertaModel
from mask_generator import FeatureMask
import torch.nn.functional as F

from . import mm_utils
from . import objectives
from .t5_model import T5ForMultimodalGeneration

torch.backends.cudnn.enabled = False


class MMTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.mask = FeatureMask(config["input_text_embed_size"])
        self.optim_mask = torch.optim.Adam(
            self.mask.parameters(), lr = config["learning_rate"], weight_decay=0
        )
        # self.mode = self.hparams.config["mode"]
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                CLIPVisionModel.from_pretrained(config["vit"])
                T5ForMultimodalGeneration.from_pretrained(config['tokenizer'])
            torch.distributed.barrier()
        self.flag = config["vit"]
        if (config["vit"] == "transformer-vit"):
            self.img_encoder = ImageEncoder()
        elif "clip" in config["vit"]:
            self.image_transformer = CLIPVisionModel.from_pretrained("/root/autodl-tmp/" + config['vit'])
            for param in self.image_transformer.parameters():
                  param.requires_grad = False
        elif config["vit"] == 'resnet-152':
            self.resnet = cv_models.resnet152(pretrained=True)
        elif config["vit"] == 'resnet-101':
            self.resnet = cv_models.resnet101(pretrained=True)
        elif config["vit"] == 'resnet-50':
            self.resnet = cv_models.resnet50(pretrained=True)
        elif config["vit"] == 'resnet-34':
            self.resnet = cv_models.resnet34(pretrained=True)
        elif config["vit"] == 'resnet-18':
            self.resnet = cv_models.resnet18(pretrained=True)
        if "resnet" in config["vit"]:
            self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
            self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
            # self.output_dim = self.resnet_encoder[7][2].conv3.out_channels
            for param in self.resnet.parameters():
                param.requires_grad = False
    #####################################################################################
        # self.image_transformer = CLIPVisionModel.from_pretrained(config["vit"])
        self.text_transformer = T5ForMultimodalGeneration.from_pretrained(
            config['tokenizer'],
            img_hsz=config["input_image_embed_size"],
        )
        # self.text_transformer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.clf = nn.Linear(config["input_text_embed_size"], 2)
        # for param in self.image_transformer.parameters():
        #      param.requires_grad = False

        mm_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load model ======================
        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        self.pred_result = {}

    def encode_image(
            self,
            image_features,
    ):
        if self.flag == "transformer-vit":
            # x = self.image_transformer(image_features)
            return self.img_encoder(image_features)
        if self.flag[0] == 'o':
            last_hidden_state = self.image_transformer(
                pixel_values=image_features,
            ).last_hidden_state
            return last_hidden_state
        return self.resnet(image_features)

    def infer(
            self,
            batch,
    ):
        text_ids = batch[f"text_ids"]
        label_ids = batch[f"labels"]
        text_masks = batch[f"text_masks"]
        image_features = batch[f"image_features"]
        image_features = self.encode_image(image_features)
        text_outputs = self.text_transformer(
            input_ids=text_ids,
            attention_mask=text_masks,
            image_ids=image_features,
        )
        text_outputs = torch.sum(text_outputs * text_masks.unsqueeze(-1), dim=1) / torch.sum(text_masks, dim=1).unsqueeze(-1)
        logits = self.clf(text_outputs)
        ret = {
            "logits": logits,
            "label_ids": label_ids,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        ret.update(self.infer(batch))
        ret.update(objectives.compute_clf(self, ret))
        return ret

    def training_step(self, batch, batch_idx):
        mm_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        # (self.mask.mask)
        mm_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        mm_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        mm_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        mm_utils.set_task(self)
        output = self(batch)

    def test_epoch_end(self, outs):
        mm_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return mm_utils.set_schedule(self)
