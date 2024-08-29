"""
File containing many input types used for matching to avoid string matching.
"""
from enum import StrEnum

class RunMode(StrEnum):
    """Mode of the program. Training trains a network, testing tests prediction performance, build builds the dataset, visualize visualizes the architecture."""
    TRAIN = 'train'
    TEST = 'test'
    BUILD = 'build'
    VISUALIZE = 'visualize'

class LossType(StrEnum):
    """Type of loss to use during training."""
    FocalLoss = 'focal_loss'
    WCE = 'wce'
    BCE = 'bce'
    F1Score = 'f1_score'
    F1ScoreDilate = 'f1_score_dilate'

class OptimizerType(StrEnum):
    """Types of optimization algorithms."""
    Adam = 'Adam'
    SGD = 'SGD'
    RMSprop = 'RMSprop'

class ModelType(StrEnum):
    """Types of models used for the network. Some models can utilize backbones."""
    DeepLabV3 = 'DeepLabV3'
    DeepCrack = 'DeepCrack'

    # Models that can utilize backbones. (see https://github.com/qubvel/segmentation_models#models-and-backbones)
    Unet = 'Unet'
    PSPNet = 'PSPNet'
    FPN = 'FPN'
    LinkNet = 'LinkNet'

class BackboneType(StrEnum):
    """
    Types of backbone models.
    (see https://github.com/qubvel/segmentation_models#models-and-backbones)
    """
    VGG16 = 'vgg16'
    VGG19 = 'vgg19'
    ResNet18 = 'resnet18'
    ResNet34 = 'resnet34'
    ResNet50 = 'resnet50'
    ResNet101 = 'resnet101'
    ResNet152 = 'resnet152'
    ResNeXt50 = 'resnext50'
    ResNeXt101 = 'resnext101'
    SE_RESNET18 = 'seresnet18'
    SE_RESNET34 = 'seresnet34'
    SE_RESNET50 = 'seresnet50'
    SE_RESNET101 = 'seresnet101'
    SE_RESNET152 = 'seresnet152'
    SE_RESNEXT50 = 'seresnext50'
    SE_RESNEXT101 = 'seresnext101'
    SENet154 = 'senet154'
    DenseNet121 = 'densenet121'
    DenseNet169 = 'densenet169'
    DenseNet201 = 'densenet201'
    InceptionV3 = 'inceptionv3'
    InceptionResNetV2 = 'inceptionresnetv2'
    MobileNet = 'mobilenet'
    MobileNetV2 = 'mobilenetv2'
    EfficientNetB0 = 'efficientnetb0'
    EfficientNetB1 = 'efficientnetb1'
    EfficientNetB2 = 'efficientnetb2'
    EfficientNetB3 = 'efficientnetb3'
    EfficientNetB4 = 'efficientnetb4'
    EfficientNetB5 = 'efficientnetb5'
    EfficientNetB6 = 'efficientnetb6'
    EfficientNetB7 = 'efficientnetb7'

class UnetWeightInitializerType(StrEnum):
    """Type of weight intializers for Unet"""
    HENormal = 'he_normal'
    GlorotUniform = 'glorot_uniform'
    RandomUniform = 'random_uniform'
