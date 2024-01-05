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
    "HLDGS?",
    "HOLDINGS",
    "international",
    "intl",
    "IP",
    "INTELLECTUAL PROPERTY",
    r"I\.?N\.?C\.?",
    r"I\.?N\.?D\.?",
    "INDUSTRY",
    "INDUSTRIES",
    "INVEST(?:MENTS)?",
    "PATENTS?",
    r"b\.?v\.?",
    r"L\.?T\.?D\.?",
    r"I\.?N\.?C\.?",
    "CORP(?:ORATION)?",
    "COMPANY",
    "consumer[ ]?care",
    r"C\.?O\.?",
    r"D\.?B\.?A",
    r"L\.?L\.?C",
    "LIMITED",
    r"P\.?L\.?C",
    r"L\.?P\.?",
    r"L\.?L\.?P\.?",
    "GMBH",
    "AB",
    "AG",
    "AGENCY",
    "APS",
    r"A\/?S",
    "ASSETS",
    "ASD",
    "AVG",
    "(?<!for |of |and )BIOLOGY?",
    "(?<!for |of |and )BIOL(?:OGICALS?)?",
    "(?<!for |of |and )BIOSCI(?:ENCES?)?",
    "(?<!for |of |and )BIOSCIENCES?",
    "BIOTECH?(?:NOLOGY)?",
    "BUS",
    "BV",
    "CA",
    "CHEM(?:ICAL)?(?:S)?",
    "COMMONWEALTH",
    "COOP",
    "CONSUMER",
    "CROPSCIENCE",
    "DEV(?:ELOPMENT)?",
    "DIAGNOSTICS?",
    "EDU(?:CATION)?",
    "ELECTRONICS?",
    "ENG",
    "ENTERP(?:RISE)?S?",
    "europharm",
    "FARM",
    "FARMA",
    "FOUND(?:ATION)?",
    "GROUP",
    "GLOBAL",
    "H",
    "(?<!for |of |and )HEALTH(?:CARE)?",
    "HEALT",
    "HIGH TECH",
    "HIGHER",
    "IDEC",
    "INT",
    "KG",
    r"k\.?k\.?",
    "LICENSING",
    "LTS",
    "LIFE[ ]?SCIENCES?",
    "MANI(?:UFACTURING)?",
    "MFG",
    "Manufacturing",
    "MATERIAL[ ]?SCIENCE",
    "MED(?:ICAL)?(?: college)?(?: of)?",
    "molecular",
    "NETWORK",
    "No {0-9}",  # No 2
    "NV",
    "OPERATIONS?",
    "PARTICIPATIONS?",
    "PARTNERSHIPS?",
    "PETROCHEMICALS?",
    "(?:bio[ -]?)?PHARMA?S?",
    "(?:bio[ -]?)?PHARMACEUTICAL?S?",
    "PHARAMACEUTICAL?S?",
    "PLANT",
    "PLC",
    "Plastics",
    "PRIVATE",
    "PROD",
    "PRODUCTS?",
    "PTE",
    "PTY",
    "(?<!for |of )R and D",
    "(?<!for |of )R&Db?",
    "RES",
    "(?<!for |of )RES & (?:TECH|DEV)",
    "(?<!for |of )RESEARCH(?: (?:&|and) DEV(?:ELOPMENT)?)?",
    "SA",
    "SCI",
    "SCIENT",
    "(?<!for |of |and )sciences?",
    "SCIENTIFIC",
    "SE",
    "SERV(?:ICES?)?",
    "SPA",
    "SYNTHELABO",
    "SYS",
    "SYST(?:EM)?S?",
    "TECH",
    "(?<!of |and )TECHNOLOG(?:Y|IES)?",
    "TECHNOLOGY SERVICES?",
    "THERAPEUTICS?",
    "therap(?:eutics|ie)s?",
]


COMPANY_SUPPRESSIONS = [
    "»",
    "«",
    "eeig",
    "THE",
    r"^\s*-",
    r"^\s*&",
]

UNIVERSITY_SUPPRESSIONS = [
    "REGENTS?",
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
    "CALISTOGA",
    "CANADA",
    "CHINA",
    "COLORADO",
    "DE",
    "DEUTSCHLAND",
    "EU",
    "FRANCE",
    "NETHERLANDS",
    "NORTH AMERICA",
    "NA",
    "INDIA",
    "IRELAND",
    "JAPAN",
    "(?:de )?M[ée]xico",
    "MA",
    "PALO ALTO",
    "SAN DIEGO",
    "SHANGHAI",
    "TAIWAN",
    "UK",
    "US",
    "USA",
]

COMPANY_MAP = {
    "abbott": "Abbott",
    "abbvie": "Abbvie",
    "allergan": "Allergan",
    "california inst of techn": "CalTech",
    "massachusetts inst technology": "Massachusetts Institute of Technology",
    "roche": "Roche",
    "biogen": "Biogen",
    "boston scient": "Boston Scientific",
    "boston scimed": "Boston Scientific",
    "lilly co eli": "Eli Lilly",
    "lilly": "Eli Lilly",
    "glaxo": "GlaxoSmithKline",
    "merck sharp & dohme": "Merck",
    "merck frosst": "Merck",
    "sinai": "Mount Sinai",
    "medtronic": "Medtronic",
    "sloan kettering": "Sloan Kettering",
    "sanofis?": "Sanofi",
    "basf": "Basf",
    "3m": "3M",
    "medical res council": "Medical Research Council",
    "mayo": "Mayo Clinic",  # FPs
    "unilever": "Unilever",
    "gen eletric": "GE",
    "ge": "GE",
    "lg": "LG",
    "nat cancer ct": "National Cancer Center",
    "samsung": "Samsung",
    "verily": "Verily",
    "isis": "Isis",
    "broad": "Broad Institute",
    "childrens medical center": "Childrens Medical Center",
    "us gov": "US Government",
    "us health": "US Government",
    "koninkl philips": "Philips",
    "koninklijke philips": "Philips",
    "max planck": "Max Planck",
    "novartis": "Novartis",
    "pfizer": "Pfizer",
    "gilead": "Gilead",
    "merck": "Merck",
    "dow": "Dow",
    "regeneron": "Regeneron",
    "takeda": "Takeda",
    "johnson & johnson": "JnJ",
    "janssen": "Janssen",
    "Johns? Hopkins?": "Johns Hopkins",
    "Mitsubishi": "Mitsubishi",
    "Dana[- ]?Farber": "Dana Farber",
    "Novo Nordisk": "Novo Nordisk",
    "Astrazeneca": "AstraZeneca",
    "Alexion": "AstraZeneca",
    "bristol[- ]?myers squibb": "Bristol-Myers Squibb",
    "Celgene": "Bristol-Myers Squibb",
    "Samsung": "Samsung",
    "ucb": "UCB",
    "r and s": "R&S Northeast",  # mostly to avoid this over-matching other assignee terms
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
