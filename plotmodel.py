import hiddenlayer as h
from models.convmixer import *
import os
os.environ["PATH"] += os.pathsep + 'E:/Program Files (x86)/Graphviz/bin/'
# Size of a patch, $p$
patch_size: int = 8
# Number of channels in patch embeddings, $h$
d_model: int = 256
# Number of [ConvMixer layers](#ConvMixerLayer) or depth, $d$
n_layers: int = 1
# Kernel size of the depth-wise convolution, $k$
kernel_size: int = 7

model = ConvMixer(ConvMixerLayer(d_model, kernel_size), n_layers,
                  PatchEmbeddings(d_model, patch_size, 1),
                  ClassificationHead(d_model))
vis_graph = h.build_graph(model, torch.zeros([1 ,1, 28, 28]))   # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
vis_graph.save("F:/LeedsDocs/Kaggle/code/model.png")   # 保存图像的路径