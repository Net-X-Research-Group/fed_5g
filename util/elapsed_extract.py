import json

with open('trials.json', 'r') as f:
    metadata = json.load(f)

metadata = metadata['runs']
trials = {}
for run in metadata:
    trials[run['run-id']] = run['elapsed']

with open('elapsed.json', 'w') as f:
    json.dump(trials, f)
