from nexport import utils

model = utils.FFNetwork()
utils.export_to_json(model, verbose=2, include_metadata=True)