LARGE_PHARMA_KEYWORDS = [
    "abbott",
    "abbvie",  # ABBVIE INC
    "amgen",  # AMGEN INC
    "allergan",
    "astrazenca",  # ASTRAZENECA AB
    "bayer",  # BAYER AG, Bayer Pharma AG
    "biogen",  # BIOGEN IDEC INC
    "boehringer ingelheim",  # BOEHRINGER INGELHEIM INT, BOEHRINGER INGELHEIM PHARMA
    "bristol[ -]?myers squibb",  # BRISTOL MYERS SQUIBB CO
    "eli lilly",  # LILLY CO ELI
    "lilly",
    "genentech",  # GENENTECH INC
    "gilead",  # GILEAD SCIENCES INC
    "glaxo",  # GLAXO GROUP LTD
    "glaxo[- ]?smith[- ]?kline",  # GLAXOSMITHKLINE IP DEV LTD
    "gsk",
    "janssen",  # JANSSEN PHARMACEUTICA NV
    "merck",  # MERCK SHARP & DOHME
    "mitsubishi",  # MITSUBISHI TANABE PHARMA CORP
    "novartis",  # NOVARTIS AG
    "novo nordisk",
    "pfizer",
    "regeneron",  # REGENERON PHARMACEUTICALS
    "roche",  # HOFFMANN LA ROCHE
    "sanofi",  # SANOFI AVENTIS DEUTSCHLAND
    "takeda",  # TAKEDA PHARMACEUTICAL
]

COMPANY_STRINGS = [
    "grp",
    "hldgs?",
    "holdings",
    "international",
    "intl",
    "ip",
    "intellectual property",
    "i\\.?n\\.?c\\.?",
    "i\\.?n\\.?d\\.?",
    "industry",
    "industries",
    "invest(?:ments)?",
    "ltda",
    "patents?",
    "b\\.?v\\.?",
    "l\\.?t\\.?d\\.?",
    "i\\.?n\\.?c\\.?",
    "corp(?:oration)?",
    "company",
    "consumer[ ]?care",
    "c\\.?o\\.?",
    "d\\.?b\\.?a",
    "l\\.?l\\.?c",
    "limited",
    "p\\.?l\\.?c",
    "l\\.?p\\.?",
    "l\\.?l\\.?p\\.?",
    "gmbh",
    "ab",
    "ag",
    "agency",
    "aps",
    "a\\/?s",
    "assets",
    "asd",
    "avg",
    "(?<!for |of |and )biology?",
    "(?<!for |of |and )biol(?:ogicals?)?",
    "(?<!for |of |and )biosci(?:ences?)?",
    "(?<!for |of |and )biosciences?",
    "biotech?(?:nology)?",
    "bus",
    "bv",
    "ca",
    "chem(?:ical)?(?:s)?",
    "commonwealth",
    "coop",
    "consumer",
    "cropscience",
    "dev(?:elopment)?",
    "diagnostics?",
    "edu(?:cation)?",
    "electronics?",
    "eng",
    "enterp(?:rise)?s?",
    "europharm",
    "farm",
    "farma",
    "found(?:ation)?",
    "group",
    "global",
    "h",
    "(?<!for |of |and )health(?:care)?",
    "healt",
    "high tech",
    "higher",
    "idec",
    "int",
    "kg",
    "k\\.?k\\.?",
    "licensing",
    "lts",
    "life[ ]?sciences?",
    "mani(?:ufacturing)?",
    "mfg",
    "manufacturing",
    "material[ ]?science",
    "med(?:ical)?(?: college)?(?: of)?",
    "molecular",
    "network",
    "no {0-9}",
    "nv",
    "operations?",
    "participations?",
    "partnerships?",
    "petrochemicals?",
    "(?:bio[ -]?)?pharma?s?",
    "(?:bio[ -]?)?pharmaceutical?s?",
    "pharamaceutical?s?",
    "plant",
    "plc",
    "plastics",
    "private",
    "prod",
    "products?",
    "pte",
    "pty",
    "(?<!for |of )r and d",
    "(?<!for |of )r&db?",
    "res",
    "(?<!for |of )res & (?:tech|dev)",
    "(?<!for |of )research(?: (?:&|and) dev(?:elopment)?)?",
    "sa",
    "sci",
    "scient",
    "(?<!for |of |and )sciences?",
    "scientific",
    "se",
    "serv(?:ices?)?",
    "spa",
    "synthelabo",
    "sys",
    "syst(?:em)?s?",
    "tech",
    "(?<!of |and )technolog(?:y|ies)?",
    "technology services?",
    "therapeutics?",
    "therap(?:eutics|ie)s?",
]


COMPANY_SUPPRESSIONS = [
    "»",
    "«",
    "eeig",
    "the",
    r"^\s*-",
    r"^\s*&",
]

UNIVERSITY_SUPPRESSIONS = [
    "regents?",
    "School Of Medicine",
    "alumni",
]

HEALTH_SYSTEM_SUPPRESSIONS = [
    "h(?:ea)?lth[ ]?care",  # e.g 'Viiv Hlthcare'
    "health[ ]?system",
]

COMPANY_INDICATORS = [
    "inc",
    "llc",
    "ltd",
    "corp",
    "corporation",
    "co",
    "gmbh",
]


COUNTRIES = [
    "calistoga",
    "canada",
    "china",
    "colorado",
    "de",
    "deutschland",
    "eu",
    "france",
    "netherlands",
    "north america",
    "na",
    "india",
    "ireland",
    "japan",
    "(?:de )?m[ée]xico",
    "ma",
    "palo alto",
    "san diego",
    "shanghai",
    "taiwan",
    "uk",
    "us",
    "usa",
]

COMPANY_MAP = {
    "abbott": "abbott",
    "abbvie": "abbvie",
    "allergan": "allergan",
    "california inst of techn": "caltech",
    "massachusetts inst technology": "massachusetts institute of technology",
    "roche": "roche",
    "biogen": "biogen",
    "boston scient": "boston scientific",
    "boston scimed": "boston scientific",
    "lilly co eli": "eli lilly",
    "lilly": "eli lilly",
    "glaxo": "glaxosmithkline",
    "smithkline beecham": "glaxosmithkline",
    "merck": "merck",
    "sinai": "mount sinai",
    "medtronic": "medtronic",
    "sloan kettering": "sloan kettering",
    "sanofis?": "sanofi",
    "basf": "basf",
    "3m": "3m",
    "medical res council": "medical research council",
    "mayo": "mayo clinic",  # FPs
    "unilever": "unilever",
    "gen eletric": "ge",
    "ge": "ge",
    "lg": "lg",
    "nat cancer ct": "national cancer center",
    "samsung": "samsung",
    "verily": "verily",
    "isis": "isis",
    "broad": "broad institute",
    "childrens medical center": "childrens medical center",
    "us gov": "us government",
    "us health": "us government",
    "koninkl philips": "philips",
    "koninklijke philips": "philips",
    "max planck": "max planck",
    "novartis": "novartis",
    "pfizer": "pfizer",
    "gilead": "gilead",
    "dow": "dow",
    "regeneron": "regeneron",
    "takeda": "takeda",
    "johnson & johnson": "jnj",
    "janssen": "janssen",
    "johns? hopkins?": "johns hopkins",
    "mitsubishi": "mitsubishi",
    "dana[- ]?farber": "dana farber",
    "novo nordisk": "novo nordisk",
    "astrazeneca": "astrazeneca",
    "alexion": "astrazeneca",
    "bristol[- ]?myers squibb": "bristol-myers squibb",
    "celgene": "bristol-myers squibb",
    "samsung": "samsung",
    "ucb": "ucb",
    "r and s": "r&s northeast",  # mostly to avoid this over-matching other assignee terms
}

OWNER_TERM_MAP = {
    "lab": "laboratory",
    "labs": "laboratories",
    "univ": "university",
    "inst": "institute",
    "govt": "government",
    "gov": "government",
    "dept": "department",
}


OWNER_SUPPRESSIONS = [
    *COMPANY_SUPPRESSIONS,
    *UNIVERSITY_SUPPRESSIONS,
    *COMPANY_STRINGS,
    # no country names unless at start of string or preceded by "of"
    *[f"(?<!^)(?<!of ){c}" for c in COUNTRIES],
    # remove stock words from company names
    r"(?:class [a-z]? )?(?:(?:[0-9](?:\.[0-9]*)% )?(?:convertible |american )?(?:common|ordinary|preferred|voting|deposit[ao]ry) (?:stock|share|sh)s?|warrants?).*",
]
