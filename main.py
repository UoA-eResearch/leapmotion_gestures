#!/usr/bin/env python3

import websocket
import config
import json
import pandas as pd

websocket_cache = {}

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

if __name__ == "__main__":
    frames = []
    try:
        while True:
            for i, device in enumerate(config.devices):
                if i not in websocket_cache:
                    ws = websocket.create_connection(device["url"])
                    if device["mode"] == "desktop":
                        ws.send(json.dumps({"optimizeHMD": False}))
                    else:
                        ws.send(json.dumps({"optimizeHMD": True}))
                    version = ws.recv()
                    print(i, version)
                    websocket_cache[i] = ws
                else:
                    resp = websocket_cache[i].recv()
                    if "event" in resp:
                        # connect / disconnect
                        print(i, resp)
                    else:
                        frame = json.loads(resp)
                        if frame["hands"]:
                            packed_frame = dict([(k,v) for k,v in frame.items() if type(v) in [int, float]])
                            packed_frame["device_index"] = i
                            packed_frame["device_mode"] = 0 if device["mode"] == "desktop" else 1
                            for hand in frame["hands"]:
                                left_or_right = hand["type"]
                                for key, value in hand.items():
                                    if key == "type":
                                        continue
                                    if key == "armBasis":
                                        #flatten
                                        value = [item for sublist in value for item in sublist]
                                    if type(value) is list:
                                        for j, v in enumerate(value):
                                            packed_frame["_".join((left_or_right, key, str(j)))] = v
                                    else:
                                        packed_frame["_".join((left_or_right, key))] = value
                            for finger in frame["pointables"]:
                                if finger["handId"] == packed_frame.get("left_id"):
                                    left_or_right = "left"
                                elif finger["handId"] == packed_frame.get("right_id"):
                                    left_or_right = "right"
                                finger_name = FINGERS[finger["type"]]
                                for key, value in finger.items():
                                    if key == "type":
                                        continue
                                    if key == "bases":
                                        #flatten
                                        value = [item for sublist in value for subsublist in sublist for item in subsublist]
                                    if key == "extended":
                                        value = int(value)
                                    if type(value) is list:
                                        for j, v in enumerate(value):
                                            packed_frame["_".join((left_or_right, finger_name, key, str(j)))] = v
                                    else:
                                        packed_frame["_".join((left_or_right, finger_name, key))] = value
                            frames.append(packed_frame)
                            if len(frames) % 100 == 0:
                                print(f"{len(frames)} frames captured")
    except KeyboardInterrupt:
        fn = input("Enter filename to save recording to: ")
        df = pd.DataFrame(frames)
        df.to_csv(f"recordings/{fn}.csv", index=False)
        print("Saved")