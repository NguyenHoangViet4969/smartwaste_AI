import json
firebase = json.load(open("firebase_key.json", "r"))
print(json.dumps(firebase).replace("\n", "\\n"))
