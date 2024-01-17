import numpy as np

GESTURE_LIST_MASTER = [
    ('pro','Hand Closing'),
    ('mid','Hand Closing'),
    ('sup','Hand Closing'),
    ('pro','Hand Opening'),
    ('mid','Hand Opening'),
    ('sup','Hand Opening'),
    ('pro','Wrist Extension'),
    ('mid','Wrist Extension'),
    ('sup','Wrist Extension'),
    ('pro','Wrist Flexion'),
    ('mid','Wrist Flexion'),
    ('sup','Wrist Flexion'),
    ('pro','Radial Deviation'),
    ('mid','Radial Deviation'),
    ('sup','Radial Deviation'),
    ('pro','Ulnar Deviation'),
    ('mid','Ulnar Deviation'),
    ('sup','Ulnar Deviation'),
    ('pro','Pinch'),
    ('mid','Pinch'),
    ('sup','Pinch'),
    ('mid','Pronation'),
    ('sup','Pronation'),
    ('pro','Supination'),
    ('mid','Supination'),
]

GESTURE_KEYS = {}
for i, (pose, gesture) in enumerate(GESTURE_LIST_MASTER):
    GESTURE_KEYS[pose+'_'+gesture] = i