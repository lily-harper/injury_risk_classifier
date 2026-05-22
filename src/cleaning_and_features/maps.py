import pandas as pd 
import numpy as np

ROAD_DESCRIP_MAP = {
    "at intersection" : "intersection",
    "intersection related" : "intersection",
    "non intersection": "non_intersection",
    "non-intersection" : "non_intersection",
    "parking lot": "parking_lat",
    "highway interchange": "interchange",
    "driveway access related": "driveway",
    "alley related": "driveway",
    "": "unknown",
    "under investigation" : "unknown"
}

LIGHT_MAP = {
    "daylight" : "daylight",
    "day light" : "daylight",
    "dark-lighted" : "dark_lighted",
    "dark_unlighted" : "dark_unlighted",
    "dawn or dusk" : "dawn_or_dusk",
    "under investigation": "unknown",
    "": "unknown"
}

ROAD_CONDITION_GROUPS = {
    "dry": [
        "dry",
    ],

    "wet": [
        "wet",
    ],

    "winter_condition": [
        "icy",
        "snowy",
        "slushy",
    ],

    "treated_winter_condition": [
        "snowy with visible icy road treatment",
        "snowy w/visible icy road treatment",
        "icy with visible icy road treatment",
        "icy w/visible icy road treatment",
        "slushy with visible icy road treatment",
        "slushy w/visible icy road treatment",
        "dry with visible icy road treatment",
        "dry w/visible icy road treatment",
        "wet with visible icy road treatment",
        "wet w/visible icy road treatment",
    ],

    "debris_or_surface_issue": [
        "muddy",
        "foreign material",
        "sand/gravel",
        "roto-milled",
    ],

    "unknown": [
        "",
        "under investigation",
    ],
}

ROAD_CONDITION_MAP = {
    raw_value: group
    for group, raw_values in ROAD_CONDITION_GROUPS.items()
    for raw_value in raw_values
}

DIRECTION_MAP = {
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
    "northeast": "northeast",
    "northwest": "northwest",
    "southeast": "southeast",
    "southwest": "southwest",
    "other": "unknown",
    "under investigation": "unknown",
    "": "unknown",
    "00": "unknown",
    "01": "unknown",
    "02": "unknown",
    "03": "unknown",
    "05": "unknown",
    "07": "unknown",
}

DIRECTION_ANGLE_MAP = {
    "north": 0,
    "northeast": 45,
    "east": 90,
    "southeast": 135,
    "south": 180,
    "southwest": 225,
    "west": 270,
    "northwest": 315,
    "unknown": np.nan,
}


# dicitonaries for the binned text columns 

VEHICLE_SIZE_MAP = {
    "bicycle_or_low_speed_vehicle": 1,
    "motorcycle_or_autocycle": 1,
    "passenger": 2,
    "suv": 3,
    "pickup_or_utility_van": 3,
    "bus": 4,
    "heavy_or_special_vehicle": 5,
    "light_rail": 5,
    "other": pd.NA,
    "unknown": pd.NA,
}

VEHICLE_TYPE_GROUPS = {
    # passenger, regular car 
    "passenger" : 
        ["passenger car/van",
         "passenger car/passenger van",
         ],
    "suv" : 
        ["suv", "suv with trailer"],
    "pickup_or_utility_van": 
        ["pickup truck/utility van",
         "pickup truck/utility van with trailer",],
    "hit_and_run" : 
        ["hit and run unknown",
         "unknown (hit and run only)",
         "under investigation"],
    "other": 
        ["other", "other vehicle type (describe in narative)",
         "0", "unk"],
    "heavy_or_special_vehicle": 
        ["vehicle over 10000 lbs",
         "medium/heavy trucks gvwr/gcwr 16,001 and over",
         "medium/heavy trucks gvwr/gcwr between 10,001 and 16,000",
         "motor home", "farm equipment", "working vehicle/equipment"],
    "motorcycle_or_autocycle" : 
        ["motorcycle",
         "autocycle"],
    "bus" : ["transit bus", "non-school bus", 
             "school bus", "school bus (all school buses)",
             "non-school bus (9 occupants or more including driver) in commerce"],
    "bicycle_or_low_speed_vehicle" : 
            ["low speed vehicle",
             "off highway vehicle/atv",
             "motorized bicycle"],
    "light_rail" : ["light rail"]
}

HUMAN_CONTRIB_GROUPS = {
    "no_apparent" : 
        ["no apparent",
         "no apparent contributing factor",
         "not observed"],

    "aggressive" : 
        ["aggressive driving"],

    "distracted" : 
        ["distracted-other", 
         "distracted/other interior",
         "distracted cellphone",
         "distracted/other exterior",
         "distracted passenger",
         "distracted radio",
         "distracted/smoking",
         "distracted/other occupant",
         "distracted/manipulating vehicle control",
         "distracted eating/drinking",
         "manipulating electronic device",
         "talking on phone/holding",
         "talking on phone/hands free"],

    "did_not_see" : 
        ["looked/did not see"],

    "unknown_or_not_observed" : 
        ["other", 
         "other factor"],

    "inexperience_or_unfamiliar" : 
        ["driver inexperience",
         "age/driver ability",
         "driver unfamiliar with area"],

    "impairment" : 
        ["dui/dwai/duid"],

    "chase" : 
        ["evading law enforcement officer"],

    "medical_or_fatigue" : ["medical",
                 "driver fatigue",
                 "illness/medical",
                 "illness",
                 "asleep at the wheel",
                 "asleep or fatigued",
                 "driver emotionally upset",
                 "physical disability"],

    "investigation_or_invalid" : 
        ["under investigation",
         "18",
         "17",
         "00",
         "06"]
}

DRIVER_ACTION_GROUPS = {
    "careless_or_loss_of_control": [
        "careless driving",
        "followed too closely",
        "spun out of control",
        "over-correcting/over-steering",
        "avoiding object in roadway",
    ],

    "reckless_or_speed_related": [
        "weaving",
        "exceed safe/posted speed",
        "reckless driving",
        "too fast for conditions",
        "speeding",
        "racing",
    ],

    "no_action": [
        "no action",
        "no contributing action",
    ],

    "lane_or_passing": [
        "lane violation",
        "changing lanes",
        "passing",
        "improper passing on right",
        "improper passing on left",
        "entering traffic way/merge",
    ],

    "failure_to_yield_or_obey_control": [
        "failed to yield row",
        "failed to stop at signal",
        "disregard stop sign",
        "disregarded stop sign",
        "disregarded other device/sign/markings",
        "disregarded other device",
        "drove wrong way",
    ],

    "backing_or_parking_movement": [
        "improper backing",
        "backing",
        "entering/leaving parked position",
    ],

    "turning": [
        "improper turn",
        "turned from wrong lane or position",
        "other improper turns",
        "making left turn",
        "making right turn",
        "making u-turn",
        "signaling violation",
    ],

    "traffic_flow_or_position": [
        "stopped in traffic",
        "impeded traffic",
        "slowing",
        "going straight",
        "parked",
    ],

    "other_or_invalid": [
        "other",
        "other contributing action",
        "17",
        "00",
        "lacking required chains",
        "na",
    ],

    "under_investigation": [
        "under investigation",
    ]
}

# flatten 

def flatten_group_map(group_map):
    """
    Convert:
        clean_group -> [raw_value_1, raw_value_2]

    Into:
        raw_value -> clean_group
    """
    return {
        raw_value: group
        for group, raw_values in group_map.items()
        for raw_value in raw_values
    }


VEHICLE_TYPE_MAP = flatten_group_map(VEHICLE_TYPE_GROUPS)
HUMAN_CONTRIB_MAP = flatten_group_map(HUMAN_CONTRIB_GROUPS)
DRIVER_ACTION_MAP = flatten_group_map(DRIVER_ACTION_GROUPS)