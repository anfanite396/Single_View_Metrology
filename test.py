from ScaleNet import *
import pandas as pd
import numpy as np

model = core.model.BaseModel()
model1 = tools.dist_init()
model2 = tools.eval.Info()
model4 = core.utils.build_optimizer(model)