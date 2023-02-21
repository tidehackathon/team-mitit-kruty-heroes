import pandas as pd
import numpy as np
from pymavlink import mavutil


COLUMNS = ['time_usec', 'time_boot_ms', 'yaw', 'pitch', 'roll', 'altitude', 'lat', 'lng', 'lon', 'relative_alt']
correction_to_c = {'cm': 0.01, 'mm': 0.001, 'us': 0.000001, 'ms': 0.001, 'degE7': 1e-7}

reader = mavutil.mavlink_connection('../sample2/sample2b.tlog')

data = []
for i in range(0, reader._count):
    message = reader.recv_match()
    if message is None:
        break
    message_type = message.to_dict()['mavpackettype']
    if message_type not in ['GPS_RAW_INT', 'ATTITUDE', 'GPS2_RAW', 'AHRS2', 'AHRS3', 'GLOBAL_POSITION_INT']:
        continue

    row = {'source': message_type}
    for key in COLUMNS:
        # Check if target key exists among of message field names.
        if list(message.fieldnames).count(key) <= 0:
            continue
        # Exception for time field.
        coef = correction_to_c.get(message.fieldunits_by_name[key], 1)
        val = getattr(message, key) * coef
        if key == 'time_usec' or key == 'time_boot_ms':
            row['time'] = val
        elif key in ['lng', 'lon']:
            row['lng'] = val
        else:
            row[key] = val

    data.append(row)

df = pd.DataFrame(data)
df.to_csv('tlogs_sample2.csv', index=False)
