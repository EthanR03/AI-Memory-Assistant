"""Canonical team registry and name resolution.

The Fact Book refers to the same club three different ways depending on
the page: a full name in club headers ("BALTIMORE RAVENS"), a location
in schedules and game logs ("Baltimore"), and a terse column header in
the statistics matrices ("Bal"). Every parser funnels through
`resolve()` so the rest of the codebase only ever sees a stable
three-letter id.
"""

# (id, location, nickname, conference, division, matrix-column abbreviation)
_TEAMS = [
    ("ARI", "Arizona", "Cardinals", "NFC", "West", "Ari"),
    ("ATL", "Atlanta", "Falcons", "NFC", "South", "Atl"),
    ("BAL", "Baltimore", "Ravens", "AFC", "North", "Bal"),
    ("BUF", "Buffalo", "Bills", "AFC", "East", "Buf"),
    ("CAR", "Carolina", "Panthers", "NFC", "South", "Car"),
    ("CHI", "Chicago", "Bears", "NFC", "North", "Chi"),
    ("CIN", "Cincinnati", "Bengals", "AFC", "North", "Cin"),
    ("CLE", "Cleveland", "Browns", "AFC", "North", "Cle"),
    ("DAL", "Dallas", "Cowboys", "NFC", "East", "Dal"),
    ("DEN", "Denver", "Broncos", "AFC", "West", "Den"),
    ("DET", "Detroit", "Lions", "NFC", "North", "Det"),
    ("GB", "Green Bay", "Packers", "NFC", "North", "GB"),
    ("HOU", "Houston", "Texans", "AFC", "South", "Hou"),
    ("IND", "Indianapolis", "Colts", "AFC", "South", "Ind"),
    ("JAX", "Jacksonville", "Jaguars", "AFC", "South", "Jac"),
    ("KC", "Kansas City", "Chiefs", "AFC", "West", "KC"),
    ("LAC", "L.A. Chargers", "Chargers", "AFC", "West", "LAC"),
    ("LAR", "L.A. Rams", "Rams", "NFC", "West", "LAR"),
    ("LV", "Las Vegas", "Raiders", "AFC", "West", "LV"),
    ("MIA", "Miami", "Dolphins", "AFC", "East", "Mia"),
    ("MIN", "Minnesota", "Vikings", "NFC", "North", "Min"),
    ("NE", "New England", "Patriots", "AFC", "East", "NE"),
    ("NO", "New Orleans", "Saints", "NFC", "South", "NO"),
    ("NYG", "N.Y. Giants", "Giants", "NFC", "East", "NYG"),
    ("NYJ", "N.Y. Jets", "Jets", "AFC", "East", "NYJ"),
    ("PHI", "Philadelphia", "Eagles", "NFC", "East", "Phi"),
    ("PIT", "Pittsburgh", "Steelers", "AFC", "North", "Pit"),
    ("SEA", "Seattle", "Seahawks", "NFC", "West", "Sea"),
    ("SF", "San Francisco", "49ers", "NFC", "West", "SF"),
    ("TB", "Tampa Bay", "Buccaneers", "NFC", "South", "TB"),
    ("TEN", "Tennessee", "Titans", "AFC", "South", "Ten"),
    ("WAS", "Washington", "Commanders", "NFC", "East", "Was"),
]

# Four clubs are printed with a short location in schedules ("L.A. Rams")
# but their full city in club headers ("LOS ANGELES RAMS"). Both spellings
# have to resolve, and `full_name` has to use the city form because that
# is what the running head on each club block says.
_CITIES = {"LAC": "Los Angeles", "LAR": "Los Angeles",
           "NYG": "New York", "NYJ": "New York"}

TEAMS = {
    t[0]: {
        "team_id": t[0], "location": t[1], "nickname": t[2],
        "city": _CITIES.get(t[0], t[1]),
        "full_name": f"{_CITIES.get(t[0], t[1])} {t[2]}",
        "conference": t[3], "division": t[4], "abbr": t[5],
    }
    for t in _TEAMS
}

# Spellings the book actually uses, mapped to the canonical id. Keys are
# normalised (lowercase, no periods, collapsed spaces) by `_norm`.
_ALIASES: dict[str, str] = {}


def _norm(name: str) -> str:
    return " ".join(name.replace(".", "").replace("’", "'").lower().split())


for _t in _TEAMS:
    _id, _loc, _nick, _conf, _div, _abbr = _t
    _city = _CITIES.get(_id, _loc)
    # A bare city is deliberately NOT an alias for the four clubs that
    # share one ("Los Angeles" is two different teams); only the full
    # city-plus-nickname form is unambiguous.
    variants = [_id, _loc, _nick, _abbr, f"{_loc} {_nick}", f"{_city} {_nick}"]
    if _id not in _CITIES:
        variants.append(_city)
    for variant in variants:
        _ALIASES[_norm(variant)] = _id

# Irregular spellings scattered through the book. The Fact Book prints
# some clubs with a hyphen in the game-log pages ("L.A.-C", "N.Y.-J"),
# uses historical city names in the all-time sections, and abbreviates
# inconsistently between the schedule grid and the statistics matrices.
_ALIASES.update({
    "la chargers": "LAC", "la-c": "LAC", "l a chargers": "LAC",
    "los angeles chargers": "LAC", "san diego": "LAC", "san diego chargers": "LAC",
    "la rams": "LAR", "la-r": "LAR", "l a rams": "LAR",
    "los angeles rams": "LAR", "st louis": "LAR", "st louis rams": "LAR",
    "ny giants": "NYG", "ny-g": "NYG", "new york giants": "NYG",
    "ny jets": "NYJ", "ny-j": "NYJ", "new york jets": "NYJ",
    "oakland": "LV", "oakland raiders": "LV", "las vegas raiders": "LV",
    "washington redskins": "WAS", "washington football team": "WAS",
    "jax": "JAX", "jac": "JAX", "gnb": "GB", "kan": "KC",
    "sfo": "SF", "tam": "TB", "nwe": "NE", "nor": "NO",
})

# Longest-first so "New England" wins over "New" when scanning a line that
# has several team names jammed together (the schedule grid does this).
_ALIAS_BY_LENGTH = sorted(_ALIASES, key=len, reverse=True)


class UnknownTeam(ValueError):
    """Raised when a name cannot be matched to one of the 32 clubs."""


def resolve(name: str, *, strict: bool = True) -> str | None:
    """Map any Fact Book spelling of a club to its canonical id."""
    key = _norm(name)
    if key in _ALIASES:
        return _ALIASES[key]
    # Trailing markers the book appends to game-log entries: "+" for a
    # neutral site, "*" for a footnote.
    key = key.rstrip("+*# ").strip()
    if key in _ALIASES:
        return _ALIASES[key]
    if strict:
        raise UnknownTeam(f"cannot resolve team name: {name!r}")
    return None


def resolve_suffix(text: str) -> tuple[str, str] | None:
    """Find the club name that a string *ends* with.

    The league schedule grid prefixes bye-week clubs onto the same text
    line as an unrelated game ("Carolina  Las Vegas ____ at New England"),
    so anchoring on the end of the segment is what separates the two.
    Returns (team_id, matched_text) or None.
    """
    key = _norm(text)
    for alias in _ALIAS_BY_LENGTH:
        if key == alias or key.endswith(" " + alias):
            return _ALIASES[alias], alias
    return None


def by_abbr(abbr: str) -> str:
    """Resolve a statistics-matrix column header ("Bal", "NYJ")."""
    return resolve(abbr)
