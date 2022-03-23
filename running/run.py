from wraped_model import VarMisuseLayer
import yaml


config = yaml.safe_load(open('../config.yml'))
model = VarMisuseLayer(config['model'], 2)
print(model.embedding)
print(model.pos_enc)
print(model.model)
