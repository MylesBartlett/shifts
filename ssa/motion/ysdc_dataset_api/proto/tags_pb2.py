# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tags.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import enum_type_wrapper

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='tags.proto',
    package='neurips_dataset',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\ntags.proto\x12\x0fneurips_dataset\"\xf5\x01\n\tSceneTags\x12-\n\x08\x64\x61y_time\x18\x01 \x01(\x0e\x32\x1b.neurips_dataset.DayTimeTag\x12*\n\x06season\x18\x02 \x01(\x0e\x32\x1a.neurips_dataset.SeasonTag\x12%\n\x05track\x18\x03 \x01(\x0e\x32\x16.neurips_dataset.Track\x12/\n\tsun_phase\x18\x04 \x01(\x0e\x32\x1c.neurips_dataset.SunPhaseTag\x12\x35\n\rprecipitation\x18\x05 \x01(\x0e\x32\x1e.neurips_dataset.Precipitation*\xb2\x01\n\rTrajectoryTag\x12\r\n\tkMoveLeft\x10\x00\x12\x0e\n\nkMoveRight\x10\x01\x12\x10\n\x0ckMoveForward\x10\x02\x12\r\n\tkMoveBack\x10\x03\x12\x11\n\rkAcceleration\x10\x04\x12\x11\n\rkDeceleration\x10\x05\x12\x0c\n\x08kUniform\x10\x06\x12\r\n\tkStopping\x10\x07\x12\r\n\tkStarting\x10\x08\x12\x0f\n\x0bkStationary\x10\t*Y\n\nDayTimeTag\x12\x13\n\x0f_kUnusedDayTime\x10\x00\x12\n\n\x06kNight\x10\x01\x12\x0c\n\x08kMorning\x10\x02\x12\x0e\n\nkAfternoon\x10\x03\x12\x0c\n\x08kEvening\x10\x04*S\n\tSeasonTag\x12\x12\n\x0e_kUnusedSeason\x10\x00\x12\x0b\n\x07kWinter\x10\x01\x12\x0b\n\x07kSpring\x10\x02\x12\x0b\n\x07kSummer\x10\x03\x12\x0b\n\x07kAutumn\x10\x04*Y\n\x0bSunPhaseTag\x12\x14\n\x10_kUnusedDaylight\x10\x00\x12\x16\n\x12kAstronomicalNight\x10\x01\x12\r\n\tkTwilight\x10\x02\x12\r\n\tkDaylight\x10\x03*j\n\x05Track\x12\x11\n\r_kUnusedTrack\x10\x00\x12\n\n\x06Moscow\x10\x01\x12\x0c\n\x08Skolkovo\x10\x02\x12\r\n\tInnopolis\x10\x03\x12\x0c\n\x08\x41nnArbor\x10\x04\x12\n\n\x06Modiin\x10\x05\x12\x0b\n\x07TelAviv\x10\x06*b\n\rPrecipitation\x12\x19\n\x15_kUnusedPrecipitation\x10\x00\x12\x14\n\x10kNoPrecipitation\x10\x01\x12\t\n\x05kRain\x10\x02\x12\n\n\x06kSleet\x10\x03\x12\t\n\x05kSnow\x10\x04\x62\x06proto3',
)

_TRAJECTORYTAG = _descriptor.EnumDescriptor(
    name='TrajectoryTag',
    full_name='neurips_dataset.TrajectoryTag',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='kMoveLeft',
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kMoveRight',
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kMoveForward',
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kMoveBack',
            index=3,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kAcceleration',
            index=4,
            number=4,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kDeceleration',
            index=5,
            number=5,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kUniform',
            index=6,
            number=6,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kStopping',
            index=7,
            number=7,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kStarting',
            index=8,
            number=8,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kStationary',
            index=9,
            number=9,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=280,
    serialized_end=458,
)
_sym_db.RegisterEnumDescriptor(_TRAJECTORYTAG)

TrajectoryTag = enum_type_wrapper.EnumTypeWrapper(_TRAJECTORYTAG)
_DAYTIMETAG = _descriptor.EnumDescriptor(
    name='DayTimeTag',
    full_name='neurips_dataset.DayTimeTag',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='_kUnusedDayTime',
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kNight',
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kMorning',
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kAfternoon',
            index=3,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kEvening',
            index=4,
            number=4,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=460,
    serialized_end=549,
)
_sym_db.RegisterEnumDescriptor(_DAYTIMETAG)

DayTimeTag = enum_type_wrapper.EnumTypeWrapper(_DAYTIMETAG)
_SEASONTAG = _descriptor.EnumDescriptor(
    name='SeasonTag',
    full_name='neurips_dataset.SeasonTag',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='_kUnusedSeason',
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kWinter',
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kSpring',
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kSummer',
            index=3,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kAutumn',
            index=4,
            number=4,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=551,
    serialized_end=634,
)
_sym_db.RegisterEnumDescriptor(_SEASONTAG)

SeasonTag = enum_type_wrapper.EnumTypeWrapper(_SEASONTAG)
_SUNPHASETAG = _descriptor.EnumDescriptor(
    name='SunPhaseTag',
    full_name='neurips_dataset.SunPhaseTag',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='_kUnusedDaylight',
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kAstronomicalNight',
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kTwilight',
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kDaylight',
            index=3,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=636,
    serialized_end=725,
)
_sym_db.RegisterEnumDescriptor(_SUNPHASETAG)

SunPhaseTag = enum_type_wrapper.EnumTypeWrapper(_SUNPHASETAG)
_TRACK = _descriptor.EnumDescriptor(
    name='Track',
    full_name='neurips_dataset.Track',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='_kUnusedTrack',
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='Moscow',
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='Skolkovo',
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='Innopolis',
            index=3,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='AnnArbor',
            index=4,
            number=4,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='Modiin',
            index=5,
            number=5,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='TelAviv',
            index=6,
            number=6,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=727,
    serialized_end=833,
)
_sym_db.RegisterEnumDescriptor(_TRACK)

Track = enum_type_wrapper.EnumTypeWrapper(_TRACK)
_PRECIPITATION = _descriptor.EnumDescriptor(
    name='Precipitation',
    full_name='neurips_dataset.Precipitation',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name='_kUnusedPrecipitation',
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kNoPrecipitation',
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kRain',
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kSleet',
            index=3,
            number=3,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name='kSnow',
            index=4,
            number=4,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=835,
    serialized_end=933,
)
_sym_db.RegisterEnumDescriptor(_PRECIPITATION)

Precipitation = enum_type_wrapper.EnumTypeWrapper(_PRECIPITATION)
kMoveLeft = 0
kMoveRight = 1
kMoveForward = 2
kMoveBack = 3
kAcceleration = 4
kDeceleration = 5
kUniform = 6
kStopping = 7
kStarting = 8
kStationary = 9
_kUnusedDayTime = 0
kNight = 1
kMorning = 2
kAfternoon = 3
kEvening = 4
_kUnusedSeason = 0
kWinter = 1
kSpring = 2
kSummer = 3
kAutumn = 4
_kUnusedDaylight = 0
kAstronomicalNight = 1
kTwilight = 2
kDaylight = 3
_kUnusedTrack = 0
Moscow = 1
Skolkovo = 2
Innopolis = 3
AnnArbor = 4
Modiin = 5
TelAviv = 6
_kUnusedPrecipitation = 0
kNoPrecipitation = 1
kRain = 2
kSleet = 3
kSnow = 4


_SCENETAGS = _descriptor.Descriptor(
    name='SceneTags',
    full_name='neurips_dataset.SceneTags',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='day_time',
            full_name='neurips_dataset.SceneTags.day_time',
            index=0,
            number=1,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='season',
            full_name='neurips_dataset.SceneTags.season',
            index=1,
            number=2,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='track',
            full_name='neurips_dataset.SceneTags.track',
            index=2,
            number=3,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='sun_phase',
            full_name='neurips_dataset.SceneTags.sun_phase',
            index=3,
            number=4,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name='precipitation',
            full_name='neurips_dataset.SceneTags.precipitation',
            index=4,
            number=5,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=32,
    serialized_end=277,
)

_SCENETAGS.fields_by_name['day_time'].enum_type = _DAYTIMETAG
_SCENETAGS.fields_by_name['season'].enum_type = _SEASONTAG
_SCENETAGS.fields_by_name['track'].enum_type = _TRACK
_SCENETAGS.fields_by_name['sun_phase'].enum_type = _SUNPHASETAG
_SCENETAGS.fields_by_name['precipitation'].enum_type = _PRECIPITATION
DESCRIPTOR.message_types_by_name['SceneTags'] = _SCENETAGS
DESCRIPTOR.enum_types_by_name['TrajectoryTag'] = _TRAJECTORYTAG
DESCRIPTOR.enum_types_by_name['DayTimeTag'] = _DAYTIMETAG
DESCRIPTOR.enum_types_by_name['SeasonTag'] = _SEASONTAG
DESCRIPTOR.enum_types_by_name['SunPhaseTag'] = _SUNPHASETAG
DESCRIPTOR.enum_types_by_name['Track'] = _TRACK
DESCRIPTOR.enum_types_by_name['Precipitation'] = _PRECIPITATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SceneTags = _reflection.GeneratedProtocolMessageType(
    'SceneTags',
    (_message.Message,),
    {
        'DESCRIPTOR': _SCENETAGS,
        '__module__': 'tags_pb2'
        # @@protoc_insertion_point(class_scope:neurips_dataset.SceneTags)
    },
)
_sym_db.RegisterMessage(SceneTags)


# @@protoc_insertion_point(module_scope)
