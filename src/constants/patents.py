from pydash import flatten
from constants.patterns.intervention import ALL_INTERVENTION_BASE_TERMS
from typings.patents import SuitabilityScoreMap
from utils.re import expand_res

BIOMEDICAL_IPC_CODE_PREFIXES = ["A61", "C01", "C07", "C08", "C12"]
BIOMEDICAL_IPC_CODE_PREFIX_RE = r"^({})".format("|".join(BIOMEDICAL_IPC_CODE_PREFIXES))


COMPOSITION_OF_MATTER_IPC_CODES = [
    "C01",  # inorganic chemistry
    "C07",  # organic chemistry
    "C08",  # organic macromolecular compounds
    "C12",  # biochemistry; beer; spirits; wine; vinegar; microbiology; enzymology; mutation or genetic engineering
]

# less likely
# "C09",  # dyes; paints; polishes; natural resins; adhesives; compositions not otherwise provided for
# "C10",  # petroleum, gas or coke industries; technical gases containing carbon monoxide; fuels; lubricants; peat
# "C11",  # ANIMAL OR VEGETABLE OILS, FATS, FATTY SUBSTANCES OR WAXES; FATTY ACIDS THEREFROM; DETERGENTS; CANDLES

METHOD_OF_USE_IPC_CODES = [
    "A01N",  # Methods of using biocides or herbicides
    "A61B",  # Methods of using surgical instruments or devices
    "A61F",  # Methods of using implants or prostheses
    "A61K",  # Preparations for medical, dental, or toilet purposes.
    "A61M",  # Methods of using catheters or other medical devices
    "A61P",  # Specific therapeutic activity of chemical compounds or medicinal preparations.
    "C07K",  # Methods of using immunoglobulins
    "C12N",  # Methods of using microorganisms or enzymes in food or beverage production
    "G01N",  # Methods of using immunoassays or other analytical techniques
    "H04N",  # Methods of using video or multimedia systems
    "H04R",  # Methods of using audio or acoustic systems
]

COMPANY_SUPPRESSIONS_DEFINITE = [
    "»",
    "«",
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
    "alumni",
    "health[ ]?care",
    "health[ ]?system",
    "HLDGS?",
    "HOLDINGS",
    "IP",
    "INTELLECTUAL PROPERTY",
    r"I\.?N\.?C\.?",
    r"I\.?N\.?D\.?",
    "INDUSTRY",
    "INDUSTRIES",
    "INVEST(?:MENTS)?",
    "PATENTS?",
    "THE",
    r"^\s*-",
    r"^\s*&",
]

COMPANY_SUPPRESSIONS_MAYBE = [
    "AGENCY",
    "APS",
    "A[/]?S",
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
    "PHARMA?",
    "PHARMACEUTICAL?S?",
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
    "REGENTS?",
    "RES",
    "(?<!for |of )RES & (?:TECH|DEV)",
    "(?<!for |of )RESEARCH(?: (?:&|and) DEV(?:ELOPMENT)?)?",
    "SA",
    "SCI",
    "SCIENT",
    "(?<!for |of |and )sciences?",
    "SCIENTIFIC",
    "School Of Medicine",
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

COMPANY_SUPPRESSIONS = [
    *COMPANY_SUPPRESSIONS_DEFINITE,
    *COMPANY_SUPPRESSIONS_MAYBE,
    *[f"(?<!of ){c}" for c in COUNTRIES],
]


COMPANY_MAP = {
    "massachusetts inst technology": "Massachusetts Institute of Technology",
    "roche": "Roche",
    "biogen": "Biogen",
    "boston scient": "Boston Scientific",
    "boston scimed": "Boston Scientific",
    "lilly co eli": "Eli Lilly",
    "eli lilly": "Eli Lilly",
    "glaxo": "GlaxoSmithKline",
    "merck sharp & dohme": "Merck",
    "merck frosst": "Merck",
    "california inst of techn": "CalTech",
    "sinai": "Mount Sinai",
    "medtronic": "Medtronic",
    "sloan kettering": "Sloan Kettering",
    "sanofi": "Sanofi",
    "sanofis": "Sanofi",
    "basf": "Basf",
    "3m": "3M",
    "abbott": "Abbott",
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
    "dow": "Dow",
    "regeneron": "Regeneron",
    "johnson & johnson": "JnJ",
    "janssen": "Janssen",
    "Johns? Hopkins?": "Johns Hopkins",
    "Mitsubishi": "Mitsubishi",
    "Dana[- ]?Farber": "Dana Farber",
    "Novo Nordisk": "Novo Nordisk",
    "Astrazeneca": "AstraZeneca",
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
        "COMBINATION": ["combo", "combination"],
        "COMPOUND_OR_MECHANISM": [
            # TODO: remove drug?
            *flatten(expand_res(ALL_INTERVENTION_BASE_TERMS)),
            "molecule",
            "receptor",
            "therapeutic",  # TODO: will probably over-match
        ],
        "DEVICE": [
            "apparatus",
            "autoinjector",
            "auto-injector",
            "cannula",
            "computer",
            "device",
            "dressing",
            "drug matrix",
            "electronic",
            "implant",
            "injector",
            "instrument",
            "needle",
            "packaging",
            "probe",
            "prosthesis",
            "scan",
            "sensor",
            "stent",
            "syringe",
            "video",
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
        "METHOD": ["procedure", "preparation method", "method of use"],
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
        "PLATFORM": ["platform"],
        "PEDIATRIC": ["pediatric", "paediatric"],
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
        "PROCESS": [
            "culture",
            "culturing",
            "manufacture",
            "preparation",
            "process",
            "synthesis",
            "system",
            "produce",
            "method of making",
            "method for producing",
            "production method",
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
