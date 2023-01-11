import nexport

model = nexport.FFNetwork()
nexport.export_to_json(model, indent=2, verbose=1, include_metadata=False)
# nexport.export_to_file(model, filename="model")