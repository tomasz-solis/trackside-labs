# Matched-Lap Bulk Extraction Diagnostics

- Built at: 2026-09-06T20:54:04.294446+00:00
- Target sessions: 44
- Loaded sessions: 24
- Error sessions: 20
- Raw rows: 3606
- Matched-pair rows: 3555
- Skipped-pair rows: 51
- Valid aggregate rows: 213

## Matched-Pair Distribution

```json
{
  "overall": {
    "count": 213,
    "min": 3.0,
    "p25": 3.0,
    "median": 10.0,
    "p75": 30.0,
    "max": 49.0
  },
  "by_session_kind_weather": [
    {
      "session_kind": "qualifying",
      "weather_bucket": "dry",
      "count": 103,
      "min": 3.0,
      "p25": 3.0,
      "median": 3.0,
      "p75": 4.0,
      "max": 6.0
    },
    {
      "session_kind": "race",
      "weather_bucket": "dry",
      "count": 110,
      "min": 9.0,
      "p25": 20.25,
      "median": 29.5,
      "p75": 37.0,
      "max": 49.0
    }
  ]
}
```

## Matched-Gap SE Distribution

```json
{
  "count": 213,
  "min": 0.026664929180443515,
  "p25": 0.09055105055578473,
  "median": 0.15351350468320407,
  "p75": 0.2890978004170142,
  "max": 1.7768265988813459
}
```

## Skip Reasons

```json
[
  {
    "session_kind": "qualifying",
    "skip_reason": "insufficient_matched_pairs",
    "count": 27
  },
  {
    "session_kind": "qualifying",
    "skip_reason": "missing_lap_time_data",
    "count": 2
  },
  {
    "session_kind": "race",
    "skip_reason": "insufficient_matched_pairs",
    "count": 10
  },
  {
    "session_kind": "race",
    "skip_reason": "missing_lap_time_data",
    "count": 9
  },
  {
    "session_kind": "race",
    "skip_reason": "no_compound_overlap",
    "count": 1
  },
  {
    "session_kind": "race",
    "skip_reason": "teammate_dnf_no_matched_laps",
    "count": 2
  }
]
```

## Zero-Observation Sessions

```json
[
  {
    "year": 2026,
    "race_name": "Italian Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "Italian Grand Prix",
    "session_kind": "qualifying"
  },
  {
    "year": 2026,
    "race_name": "Spanish Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "Spanish Grand Prix",
    "session_kind": "qualifying"
  },
  {
    "year": 2026,
    "race_name": "Azerbaijan Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "Azerbaijan Grand Prix",
    "session_kind": "qualifying"
  },
  {
    "year": 2026,
    "race_name": "Singapore Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "Singapore Grand Prix",
    "session_kind": "qualifying"
  },
  {
    "year": 2026,
    "race_name": "United States Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "United States Grand Prix",
    "session_kind": "qualifying"
  },
  {
    "year": 2026,
    "race_name": "Mexico City Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "Mexico City Grand Prix",
    "session_kind": "qualifying"
  },
  {
    "year": 2026,
    "race_name": "S\u00e3o Paulo Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "S\u00e3o Paulo Grand Prix",
    "session_kind": "qualifying"
  },
  {
    "year": 2026,
    "race_name": "Las Vegas Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "Las Vegas Grand Prix",
    "session_kind": "qualifying"
  },
  {
    "year": 2026,
    "race_name": "Qatar Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "Qatar Grand Prix",
    "session_kind": "qualifying"
  },
  {
    "year": 2026,
    "race_name": "Abu Dhabi Grand Prix",
    "session_kind": "race"
  },
  {
    "year": 2026,
    "race_name": "Abu Dhabi Grand Prix",
    "session_kind": "qualifying"
  }
]
```

## Weather Buckets

```json
[
  {
    "session_kind": "qualifying",
    "weather_bucket": "dry",
    "count": 363
  },
  {
    "session_kind": "qualifying",
    "weather_bucket": "wet",
    "count": 2
  },
  {
    "session_kind": "race",
    "weather_bucket": "dry",
    "count": 3180
  },
  {
    "session_kind": "race",
    "weather_bucket": "wet",
    "count": 10
  }
]
```

## Connected Components

```json
[
  {
    "session_kind": "qualifying",
    "weather_bucket": "dry",
    "n_components": 10,
    "component_sizes": [
      5,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2
    ],
    "component_observation_counts": [
      20,
      9,
      5,
      12,
      8,
      10,
      5,
      11,
      11,
      12
    ],
    "component_observation_shares": [
      0.1941747572815534,
      0.08737864077669903,
      0.04854368932038835,
      0.11650485436893204,
      0.07766990291262135,
      0.0970873786407767,
      0.04854368932038835,
      0.10679611650485436,
      0.10679611650485436,
      0.11650485436893204
    ],
    "components": [
      [
        "HAD",
        "LAW",
        "LIN",
        "TSU",
        "VER"
      ],
      [
        "ALB",
        "SAI"
      ],
      [
        "ALO",
        "STR"
      ],
      [
        "ANT",
        "RUS"
      ],
      [
        "BEA",
        "OCO"
      ],
      [
        "BOR",
        "HUL"
      ],
      [
        "BOT",
        "PER"
      ],
      [
        "COL",
        "GAS"
      ],
      [
        "HAM",
        "LEC"
      ],
      [
        "NOR",
        "PIA"
      ]
    ]
  },
  {
    "session_kind": "race",
    "weather_bucket": "dry",
    "n_components": 11,
    "component_sizes": [
      3,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2
    ],
    "component_observation_counts": [
      10,
      9,
      9,
      11,
      11,
      9,
      10,
      11,
      8,
      12,
      10
    ],
    "component_observation_shares": [
      0.09090909090909091,
      0.08181818181818182,
      0.08181818181818182,
      0.1,
      0.1,
      0.08181818181818182,
      0.09090909090909091,
      0.1,
      0.07272727272727272,
      0.10909090909090909,
      0.09090909090909091
    ],
    "components": [
      [
        "LAW",
        "LIN",
        "TSU"
      ],
      [
        "ALB",
        "SAI"
      ],
      [
        "ALO",
        "STR"
      ],
      [
        "ANT",
        "RUS"
      ],
      [
        "BEA",
        "OCO"
      ],
      [
        "BOR",
        "HUL"
      ],
      [
        "BOT",
        "PER"
      ],
      [
        "COL",
        "GAS"
      ],
      [
        "HAD",
        "VER"
      ],
      [
        "HAM",
        "LEC"
      ],
      [
        "NOR",
        "PIA"
      ]
    ]
  }
]
```

## Extraction Errors

```json
[
  {
    "year": 2026,
    "race_name": "Italian Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Italian Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Spanish Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Spanish Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Azerbaijan Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Azerbaijan Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Singapore Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Singapore Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "United States Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "United States Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Mexico City Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Mexico City Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "S\u00e3o Paulo Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "S\u00e3o Paulo Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Las Vegas Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Las Vegas Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Qatar Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Qatar Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Abu Dhabi Grand Prix",
    "session_kind": "race",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  },
  {
    "year": 2026,
    "race_name": "Abu Dhabi Grand Prix",
    "session_kind": "qualifying",
    "error_type": "DataNotLoadedError",
    "message": "The data you are trying to access has not been loaded yet. See `Session.load`"
  }
]
```
