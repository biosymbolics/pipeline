OTHER_OWNER_NAME = "other"
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


UNHELPFUL_COMPANY_ACROYNMS = [
    "ab",
    "ag",
    "as",
    "a/s",
    "bv",
    "b v",
    "dba",
    "gmbh",
    "int",
    "ind",
    "intl",
    "ip",
    "kg",
    "kk",
    "ltda",
    "nv",
    "n v",
    "sa",
    "s a",
    "sl",
    "s l",
]

COMMON_COMPANY_ACRONYMS = [
    *UNHELPFUL_COMPANY_ACROYNMS,
    "co",
    "corp",
    "inc",
    "int",
    "llc",
    "llp",
    "lp",
    "ltd",
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
    "center",  # also "cancer center"
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
    "invest",
    "investig",
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
    "inst",
    "institut",
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

COMMON_NON_PROFIT_WORDS = [
    "alliance",
    "council",
]

COMMON_FOUNDATION_WORDS = [
    "found",
    "foundation",
    "trust",
]


COMMON_HEALTH_SYSTEM_WORDS = [
    "healthcare",
    "(?:medical|cancer|health) (?:center|centre|system|hospital)s?",
    "clinics?",
    "districts?",
    "health",
    "healthcare",
    "hospital",
]

COMMON_OWNER_WORDS = [
    *COMMON_COMPANY_WORDS,
    *COMMON_UNIVERSITY_WORDS,
    *COMMON_NON_PROFIT_WORDS,
    *COMMON_FOUNDATION_WORDS,
    *COMMON_HEALTH_SYSTEM_WORDS,
    "animal",
    "center",
    "centre",
    "care",
    "department",
    "fund",
    "human",
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


COMMON_INDIVIDUAL_WORDS = [r"m\.?d\.?", r"d\.?r\\.?", r"ph\.?d\.?"]

# not good to suppress
COMMON_GOVT_WORDS = [
    "city",
    "government",
    "govt",
    "gov",
    "federal",
    "national",
    "governing",
    "commonwealth",
    "state",
    "us health",
    "veterans affairs",
    "nih",
    "nat",
    "va",
    "commissariat",
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
    # "alcon": "novartis", # split from novartis in 2019
    "actelion": "johnson & johnson",
    "forest laboratory": "abbvie",  # acquired by allergan, which was acquired by abbvie
    "hospira": "pfizer",
    "sigma[- ]?aldrich": "merck",
    "immunex": "amgen",
    "medimmune": "astrazeneca",
    "serono": "merck",
    "kite pharma": "gilead",
    "bioverativ": "sanofi",
    "onyx": "amgen",
    "allergan": "abbvie",
    # "catalent": "novo nordisk",
    "wyeth": "pfizer",
    "genentech": "roche",
}

COMPANY_MAP = {
    **MA_MAPPINGS,
    "abbott": "abbott",
    "abbvie": "abbvie",
    "amgen": "amgen",
    "bayer": "bayer",  # bayer agrochem
    "boehringer ingelheim": "boehringer ingelheim",
    "california inst of techn": "caltech",
    "du pont": "dupont",
    "massachusetts inst technology": "massachusetts institute of technology",
    "roche": "roche",
    "biogen": "biogen",
    "boston scient": "boston scientific",
    "boston scimed": "boston scientific",
    "lilly co eli": "eli lilly",
    "lilly": "eli lilly",
    "glaxo": "glaxosmithkline",
    "glaxosmithkline": "glaxosmithkline",  # e.g.  glaxosmithkline biologicals
    "merck": "merck",
    "sinai": "mount sinai",
    "medtronic": "medtronic",
    "kettering": "sloan kettering",
    "sanofis?": "sanofi",
    "basf": "basf",
    "exxonmobil": "exxonmobil",
    "3m": "3m",
    "matsushita": "matsushita",
    "medical res council": "medical research council",
    "mayo": "mayo clinic",  # FPs
    "unilever": "unilever",
    "general electric": "general electric",
    "gen eletric": "general electric",
    "ge healthcare": "general electric",
    "lg": "lg",
    "nat cancer ct": "national cancer center",
    "samsung": "samsung",
    "verily": "verily",
    "isis": "isis",
    "broad": "broad institute",
    "childrens medical center": "childrens medical center",
    "us gov": "us government",
    "us health": "us government",
    "us agriculture": "us government",
    "united states government": "us government",
    "koninkl philips": "philips",
    "koninklijke philips": "philips",
    "max planck": "max planck",
    "novartis": "novartis",
    "pfizer": "pfizer",
    "gilead": "gilead",
    "dow": "dow inc",
    "procter .{1,4} gamble": "procter and gamble",
    "regeneron": "regeneron",
    "takeda": "takeda",
    "jnj": "johnson & johnson",
    "johnson & johnson": "johnson & johnson",
    "janssen": "johnson & johnson",
    "johns? hopkins?": "johns hopkins",
    "mitsubishi": "mitsubishi",
    "dana[- ]?farber": "dana farber",
    "novo nordisk": "novo nordisk",
    "astrazeneca": "astrazeneca",
    "alexion": "astrazeneca",
    "bristol[- ]?myers squibb": "bristol-myers squibb",
    "samsung": "samsung",
    "ucb": "ucb",
    "syngenta": "syngenta",
    "teva": "teva",
    "uop": "honeywell",
    "kimberly clark": "kimberly clark",
    "dainippon": "dainippon",
    "scimed life": "scimed life",
    "fujifilm": "fujifilm",
    "olympus": "olympus",
    "sumitomo": "sumitomo",
    "siemens": "siemens",
    "national cancer institute": "national cancer institute",
    "nci": "national cancer institute",
    "centre nat rech scient": "centre national de la recherche scientifique",
    "cnrs": "centre national de la recherche scientifique",
    "harvard": "harvard",
    "u.s. army": "us army",
    "united states army": "us army",
    "us dept veterans affairs": "us veterans affairs",
    "sun pharmaceutical": "sun pharmaceutical",
    "sun pharma": "sun pharmaceutical",
    "sun chemical": "sun chemical",
    "advanced cardiovascular system": "advanced cardiovascular systems",
    "va office": "veterans affairs",
    "nyu": "new york university",
    "national institutes of health": "national institutes of health",
    "nih": "national institutes of health",
    "walter reed": "walter reed",
    "oxford university": "oxford university",
    "gore enterprise holdings inc": "gore",
    "rice university": "rice university",
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
    "corporation": "corp",
    "chem": "chemical",
    "hosp": "hospital",
    "limited": "ltd",
    "instr": "instrument",
    "incorporated": "inc",
    "company": "co",
    "dept?": "department",
    "depart[a-z]*": "department",
    "(?:bio)?pharm[a-z]*": "pharmaceutical",
    "r[ ]?(?:&|and)[ ]?d": "research and development",
    "res": "research",
    "dev": "development",
    "eng": "engineering",
    "biotech": "biotechnology",
    "bio[- ]*tech(?:nology)?": "biotechnology",
    "bio[- ]*pharmaceuticals?": "biopharmaceutical",
    "biotechnologies": "biotechnology",
    "techn?": "technology",
    "hldgs?": "holdings",
    "grp": "group",
    "serv": "services",
    "found": "foundation",
    "mfg": "manufacturing",
    "scienc": "science",
    "scient": "scientific",
}


OWNER_SUPPRESSIONS = [
    *UNHELPFUL_COMPANY_ACROYNMS,
    "internat[a-z]*",
    r"(?:class [a-z]? )?(?:(?:[0-9](?:\.[0-9]*)% )?(?:convertible |american )?(?:common|ordinary|preferred|voting|deposit[ao]ry) (?:stock|share|sh)s?|warrants?).*",
]
