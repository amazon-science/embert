from allennlp.data import Vocabulary
from allennlp.models import Model
from pytorch_lightning import LightningModule

from grolp.models.config import EmbodiedBertConfig
from grolp.models.embodied_bert import EmbodiedBertForPreTraining


@Model.register("embert_pretrain")
class EmbodiedBertForPretraining(Model, LightningModule):
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model_path: str,
                 use_vlmm_loss: bool = True,
                 use_vlm_loss: bool = True,
                 use_itm_loss: bool = False
                 ):
        super().__init__(vocab)
        self.optimizer = None
        self.lr_scheduler = None
        self.config: EmbodiedBertConfig = EmbodiedBertConfig(
            use_lm_loss=use_vlmm_loss,
            use_vm_loss=use_vlm_loss,
            use_itm_loss=use_itm_loss
        )
        self.save_hyperparameters(self.config.to_dict())
        self.embert = EmbodiedBertForPreTraining.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            config=self.config
        )

    def configure_optimizers(self):
        optim = {"optimizer": self.optimizer}
        if self.lr_scheduler.is_available:
            optim["lr_scheduler"] = self.lr_scheduler

        return optim

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)

        for k, v in outputs.items():
            if "loss" in k:
                self.log(k, v, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": outputs["loss"]}

    def _inference_step(self, batch, split):
        outputs = self.forward(**batch)

        for k, v in outputs.items():
            if "loss" in k:
                self.log(f"{split}_{k}", v, on_epoch=True, prog_bar=k == "loss")

        return {"loss": outputs["loss"]}

    def validation_step(self, batch, batch_idx):
        return self._inference_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._inference_step(batch, "test")

    def forward(self,
                metadata,
                token_ids,
                token_type_ids,
                text_mask,
                visual_features,
                visual_attention_mask=None,
                lang_labels=None,
                visual_labels=None):

        return self.embert(token_ids,
                           token_type_ids,
                           text_mask,
                           visual_features,
                           visual_attention_mask,
                           lang_labels,
                           visual_labels)
