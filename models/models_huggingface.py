# from torch import nn
# from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerImageProcessor
#
# class HFSegformerCityscapes(nn.Module):
#     def __init__(self, model_name, *args, **kwargs):
#         super().__init__()
#         if model_name == "segformer-b4-finetuned-cityscapes-1024-1024":
#             self.feature_extractor = SegformerFeatureExtractor.from_pretrained(
#                 "nvidia/segformer-b4-finetuned-cityscapes-1024-1024", device_map="auto")
#             self.model = SegformerForSemanticSegmentation.from_pretrained(
#                 "nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
#
#     def forward(self, x):
#         inputs = self.feature_extractor(images=x, return_tensors="pt").to("cuda")
#         outputs = self.model(**inputs)
#         logits = outputs.logits
#
#         return logits
