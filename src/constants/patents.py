from pydash import flatten

from constants.company import COMPANY_STRINGS
from constants.patterns.device import HIGH_LIKELIHOOD_DEVICES
from constants.patterns.intervention import ALL_INTERVENTION_BASE_TERMS
from typings.patents import SuitabilityScoreMap

BIOMEDICAL_IPC_CODE_PREFIXES = ["A61", "C01", "C07", "C08", "C12"]
BIOMEDICAL_IPC_CODE_PREFIX_RE = r"^({})".format("|".join(BIOMEDICAL_IPC_CODE_PREFIXES))


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

OWNER_SUPPRESSIONS = [
    *COMPANY_SUPPRESSIONS,
    *UNIVERSITY_SUPPRESSIONS,
    *COMPANY_STRINGS,
    # no country names unless at start of string or preceded by "of"
    *[f"(?<!^)(?<!of ){c}" for c in COUNTRIES],
    # remove stock words from company names
    r"(?:class [a-z]? )?(?:(?:[0-9](?:\.[0-9]*)% )?(?:convertible |american )?(?:common|ordinary|preferred|voting|deposit[ao]ry) (?:stock|share|sh)s?|warrants?).*",
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


def get_patent_attribute_map():
    return {
        "CAM": [
            "herbal",
            "herb",
            "botanical",
            "traditional medicine",
            "chinese medicine",
            "tibetian medicine",
        ],
        "COMBINATION": ["combo of", "combination of"],
        "COMPOUND_OR_MECHANISM": [
            # TODO: remove drug?
            *flatten(ALL_INTERVENTION_BASE_TERMS),
            "molecule",
            "receptor",
            "therapeutic",  # TODO: will probably over-match
        ],
        "DEVICE": [
            *HIGH_LIKELIHOOD_DEVICES,
        ],
        "DIAGNOSTIC": [
            "analysis",
            "analyte",
            "biomaker",  # IRL
            "biomarker",
            "biometric",
            "biopsy",
            "characterize",
            "diagnosis",
            "diagnostic",
            "diagnose",
            "detection",
            "evaluating response",
            "imaging",
            "immunoassay",
            "in vitro",
            "in vivo",
            "marker",
            "detecting",
            "detection",
            "monitoring",
            "mouse",
            "predict",
            "prognosis",
            "prognostic",  # needed?
            "reagent",
            "risk score",
            "scan",
            "sensor",
            "testing",
        ],
        "DISEASE_MODIFYING": [
            "disease modifying",
            "disease-modifying",
            "cure",
            "curative",
        ],
        "FORMULATION": [
            "formulation",
            "form",
            "salt",
            "extended release",
            "topical",
            "aerosol",
            "analog",
            "analogue",
        ],
        "INCREMENTAL": ["improved process", "improved method", "improved system"],
        "IRRELEVANT": [
            "mammal",
            "veterinary",
            "animal",
            "dental",
            "primate",
            "agricultural",
            "horticultural",
            "confection(?:ery)?",
            "cosmetic",
            "personal care",
            "fungicide",
            "herbicide",
            "pesticide",
            "plant disease",
            "pest control",
            "machine learning",
        ],
        "METHOD_OF_USE": ["method of use", "methods of their use"],  # see PROCESS
        "METHOD_OF_ADMINISTRATION": [
            "delivery",
            "dosing",
            "regimen",
            "administration",
        ],
        "NOVEL": ["novel", "new"],
        "NUTRITIONAL": [
            "nutrition",
            "nutritional",
            "nutraceutical",
            "vitamin",
            "dietary",
            "diet",
            "food",
            "nutrient",
            "nutriment",
        ],
        "PALLIATIVE": [
            "palliative",
            "reduction in symptoms",
            "supportive",
            "symptom management",
            "symptom relief",
            "symptom relief",
            "quality of life",
        ],
        "PEDIATRIC": ["pediatric", "paediatric"],
        "PLATFORM": ["platform"],
        "PROCESS": [
            "process(?:es)? for preparing",
            "process(?:es)? to produce",
            "process(?:es)? to make",
            "methods? of making",
            "methods? for producing",
            "culture",
            "culturing",
            "manufacture",
            "preparation",
            "procedure",
            "synthesis",
            "system for",
            "preparation method",
            "production method",
        ],
        "OBSTETRIC": [
            "obstetric",
            "obstetrical",
            "pregnant",
            "pregnancy",
            "natal",
            "neonatal",
            "fetal",
            "fetus",
        ],
        "GERIATRIC": ["geriatric", "elderly", "senior", "older adult", "older patient"],
        "PREVENTATIVE": ["prevention", "prophylaxis", "prophylactic"],
        "PROCEDURE": [
            "procedure",
            "surgery",
            "surgical",
        ],
        "TREATMENT": [
            "treatment",
            "therapeutic",
            "therapy",
        ],
    }


SUITABILITY_SCORE_MAP: SuitabilityScoreMap = {
    "CAM": -3,
    "COMBINATION": -1,
    "COMPOUND_OR_MECHANISM": 3,
    "DEVICE": -3,
    "DIAGNOSTIC": -2,
    "DISEASE_MODIFYING": 1.5,
    "FORMULATION": -1,
    "GERIATRIC": 0,
    "IRRELEVANT": -5,
    "METHOD": -1.0,  # really only a prob if not COM
    "METHOD_OF_ADMINISTRATION": -1.0,
    "NOVEL": 1.5,
    "NUTRITIONAL": -4,
    "OBSTETRIC": 0,
    "PALLIATIVE": 0,
    "PEDIATRIC": 0,
    "PLATFORM": 0,
    "PREVENTATIVE": 0,
    "PROCEDURE": -3,
    "PROCESS": -2,
    "TREATMENT": 1,
}


ATTRIBUTE_FIELD = "attributes"
