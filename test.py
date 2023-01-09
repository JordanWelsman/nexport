import nexport

model = nexport.FFNetwork()
nexport.export_to_json(model, verbose=1, include_metadata=False)