"""
Maps Retrosheet's team codes to the codes Statcast uses, so team-season
rows from both sources can be joined correctly.

Checked with check_team_codes.py: for 2015-2024,
Retrosheet uses OAK for the Athletics every year -- it does NOT
retroactively relabel history. Starting in 2025 (the team's real-world
relocation), Retrosheet switches to ATH. Statcast, by contrast, uses ATH
for every year, including ones well before the actual move.

Because of that, extending the panel through 2025 does NOT require
season-conditional logic: both OAK and ATH just need to map to
Statcast's ATH, and this stays a flat dictionary. This is only true
because both codes converge on the same Statcast target -- if Statcast
ever changes ITS code too, this would need revisiting.
"""

RETROSHEET_TO_STATCAST = {
    "ANA": "LAA",
    "ARI": "AZ",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHA": "CWS",
    "CHN": "CHC",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KCA": "KC",
    "LAN": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYA": "NYY",
    "NYN": "NYM",
    "OAK": "ATH",  # Retrosheet: OAK through 2024, ATH from 2025 -- both map here
    "ATH": "ATH",  # Retrosheet's own code starting 2025; identity mapping
    "PHI": "PHI",
    "PIT": "PIT",
    "SDN": "SD",
    "SEA": "SEA",
    "SFN": "SF",
    "SLN": "STL",
    "TBA": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WAS": "WSH",
}

# Team placeholder codes that appear in Retrosheet's batteam/pitteam
# fields but aren't real franchises (All-Star team rosters). Already
# filtered out at the source in scripts/pull_retrosheet.py, but kept
# here too as a second line of defense for anything reading raw
# Retrosheet data directly.
NON_FRANCHISE_CODES = {"ALS", "NLS"}


def to_statcast_code(retrosheet_code: str) -> str:
    """
    Convert a single Retrosheet team code to its Statcast equivalent.
    Raises KeyError with a clear message if the code isn't recognized --
    better to fail loudly here than silently drop a team's rows in a join.
    """
    try:
        return RETROSHEET_TO_STATCAST[retrosheet_code]
    except KeyError:
        raise KeyError(
            f"Unrecognized Retrosheet team code: '{retrosheet_code}'. "
            "If this is a new expansion team, a relocation, or a season "
            "outside 2015-2024, the mapping needs to be updated"
        )