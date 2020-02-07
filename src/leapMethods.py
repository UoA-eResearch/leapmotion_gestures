import config
import websocket
import json

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

def collect_frame(frames_total, n, websocket_cache):
    """Collects and unpacks a frame from a leap motion device

    Arguments:
    frames_total -- total number of frames collected so far
    n -- collect every nth frame, if frames_total % n == 0 then the frame will not be unpacked, and None will be returned
    websocket_cache -- dictionary of websocket connections, which may be updated in place with new connections

    Returns:
    packed_frame -- empty dictionary if frames_total % n == 0, else a dictionary containing variable name / value pairs


    Notes:
    This function assumes that a frame is being collected for only one device!
    """

    packed_frame = {}
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
        if frames_total % n != 0:
            # collect the frame, but don't unpack it
            resp = websocket_cache[i].recv()
        else:
            # collect and unpack
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
                    packed_frame["currentFrameRate"] /= n #adjust frame rate by the number we're actually capturing at the moment 

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
    return packed_frame