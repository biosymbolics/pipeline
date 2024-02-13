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


COMMON_COMPANY_ACRONYMS = [
    "ab",
    "ag",
    "as",
    "bv",
    "b v",
    "co",
    "corp",
    "dba",
    "gmbh",
    "inc",
    "int",
    "ind",
    "intl",
    "ip",
    "kg",
    "kk",
    "llc",
    "llp",
    "lp",
    "ltd",
    "ltda",
    "nv",
    "n v",
    "sa",
    "s a",
]


COMMON_COMPANY_WORDS = [
    *COMMON_COMPANY_ACRONYMS,
    "adv",
    "advanced",
    "agency",
    "agencies",
    "associate",
    "biotechnology",
    "biotechnologies",
    "business",
    "center",
    "chemical",
    "coop",
    "cooperative",
    "collaboration",
    "company",
    "companies",
    "consumer",
    "corporation",
    "dev",
    "development",
    "diagnostic",
    "electronic",
    "eu",
    "global",
    "group",
    "industries",
    "industry",
    "innovation",
    "intellectual",
    "international",
    "lab",
    "laboratory",
    "laboratories",
    "limited",
    "ma",
    "management",
    "manufacturing",
    "medicine",
    "network",
    "north america",
    "nutraceutical",
    "partnership",
    "pharm",
    "pharma",
    "petrochemical",
    "pharmaceutical",
    "property",
    "properties",
    "services",
    "system",
    "tech",
    "techn",
    "technology",
    "technologies",
    "therapeutic",
    "venture",
]

COMMON_UNIVERSITY_WORDS = [
    "alumni",
    "college",
    "education",
    "health",
    "healthcare",
    "hospital",
    "inst",
    "institute",
    "nyu",
    "school",
    "medicine",
    "regent",
    "univ",
    "university",
    "universities",
    "université",
]

COMMON_OWNER_WORDS = [
    *COMMON_COMPANY_WORDS,
    *COMMON_UNIVERSITY_WORDS,
    "animal",
    "center",
    "centre",
    "care",
    "commonwealth",
    "commissariat",
    "council",
    "governing",
    "department",
    "fund",
    "found",
    "foundation",
    "human",
    "invest",
    "investig",
    "medical",
    "publique",
    "res",
    "research",
    "science",
    "and",  # e.g. "bob and sons"
    "&",
]

# a little messy
PLURAL_COMMON_OWNER_WORDS = set(
    [
        *COMMON_OWNER_WORDS,
        *map(lambda x: x + "s", COMMON_OWNER_WORDS),
    ]
)

# not good to suppress
COMMON_GOVT_WORDS = [
    "city",
    "government",
    "govt",
    "federal",
    "national",
    "state",
    "us health",
    "veterans affairs",
    "nih",
    "nat",
    "va",
    "nasa",
    "prefecture",
    "european organisation",
    "eortc",
    "assistance publique",
    "fda",
    "bureau",
    "authority",
    "authorities",
]

# https://en.wikipedia.org/wiki/List_of_largest_pharmaceutical_mergers_and_acquisitions
MA_MAPPINGS = {
    "genzyme": "sanofi",
    "smithkline beecham": "glaxosmithkline",
    "celgene": "bristol-myers squibb",
    "aventis": "sanofi",
    "monsanto": "bayer",
    "covidien": "medtronic",
    "pharmacia": "pfizer",
    "shire": "takeda",
    "alcon": "novartis",
    "actelion": "johnson & johnson",
    "forest laboratory": "allergan",
    "hospira": "pfizer",
    "sigma[- ]?aldrich": "merck",
    "immunex": "amgen",
    "medimmune": "astrazeneca",
    "serono": "merck",
    "kite pharma": "gilead",
    "bioverativ": "sanofi",
    "onyx": "amgen",
    "allergan": "abbvie",
    "catalent": "novo nordisk",
}

COMPANY_MAP = {
    **MA_MAPPINGS,
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
    "merck": "merck",
    "sinai": "mount sinai",
    "medtronic": "medtronic",
    "sloan kettering": "sloan kettering",
    "sanofis?": "sanofi",
    "genzyme": "sanofi",
    "basf": "basf",
    "3m": "3m",
    "matsushita": "matsushita",
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
    "wyeth": "pfizer",
    "genentech": "genentech",
    "gilead": "gilead",
    "dow": "dow chemical",
    "procter .{1,4} gamble": "procter and gamble",
    "regeneron": "regeneron",
    "takeda": "takeda",
    "jnj": "johnson & johnson",
    "johnson & johnson": "johnson & johnson",
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
}

# can be regex. \b is added to the start and end of each term
OWNER_TERM_NORMALIZATION_MAP = {
    "labs?": "laboratory",
    "laboratories": "laboratory",
    "univ": "university",
    "inst": "institute",
    "govt?": "government",
    "nat": "national",
    "u[ .]?s[ .]?(?:a[.]?)?": "united states",
    "dept?": "department",
    "depart[a-z]*": "department",
    "(?:bio)?pharm[a-z]*": "pharmaceutical",
    "r[ ]?(?:&|and)[ ]?d": "research and development",
    "res": "research",
    "dev": "development",
    "biotech": "biotechnology",
    "techn?": "technology",
    "hldgs?": "holdings",
    "grp": "group",
    "serv": "services",
    "found": "foundation",
    "mfg": "manufacturing",
}


OWNER_SUPPRESSIONS = [
    *COMMON_COMPANY_ACRONYMS,
    "internat[a-z]*",
    "limited",
    "incorporated",
    "corporation",
    "company",
    r"(?:class [a-z]? )?(?:(?:[0-9](?:\.[0-9]*)% )?(?:convertible |american )?(?:common|ordinary|preferred|voting|deposit[ao]ry) (?:stock|share|sh)s?|warrants?).*",
]
