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
    "LTD",
    "INC",
    "CORP",
    "CORPORATION",
    "COMPANY",
    "CO",
    "DBA",
    "LLC",
    "LIMITED",
    "PLC",
    "LP",
    "LLP",
    "GMBH",
    "AB",
    "AG",
    "HLDG",
    "HLDGS",
    "HOLDINGS",
    "IP",
    "INTELLECTUAL PROPERTY",
    "INC",
    "IND",
    "INDUSTRY",
    "INDUSTRIES",
    "INVEST",
    "INVESTMENTS",
    "PATENT",
    "PATENTS",
    "THE",
    "OF$",
    r"^\s*-",
    r"^\s*&",
]

COMPANY_SUPPRESSIONS_MAYBE = [
    "AGENCY",
    "APS",
    "AS",
    "ASSETS",
    "ASD",
    "BIOLOG",
    "BIOL",
    "BIOLOGICAL",
    "BIOLOGICALS",
    "BIOSCI",
    "BIOSCIENCE",
    "BIOSCIENCES",
    "BIOTEC",
    "BIOTECH",
    "BIOTECHNOLOGY",
    "BUS",
    "BV",
    "CA",
    "CARE",
    "CHEM",
    "CHEMICAL",
    "CHEMICALS",
    "COMMONWEALTH",
    "COOP",
    "CONSUMER",
    "CROPSCIENCE",
    "DEV",
    "DEVELOPMENT",
    "DIAGNOSTIC",
    "DIAGNOSTICS",
    "EDUCATION",
    "ELECTRIC",
    "ELECTRONICS",
    "ENG",
    "ENTPR",
    "ENTERPRISE",
    "ENTERPRISES",
    "FARM",
    "FARMA",
    "FOUND",
    "FOUNDATION",
    "GROUP",
    "GLOBAL",
    "H",
    "HEALTHCARE",
    "HEALTH",
    "HEALT",
    "HIGH TECH",
    "HIGHER",
    "IDEC",
    "INT",
    "KG",
    "KK",
    "LICENSING",
    "LTS",
    "LIFESCIENCES",
    "OF$",
    "MAN",
    "MANUFACTURING",
    "MFG",
    "MATERIAL SCIENCE",
    "MATERIALSCIENCE",
    "MEDICAL",
    "MED",
    "NETWORK",
    "No {0-9}",  # No 2
    "NV",
    "OPERATIONS",
    "PARTICIPATIONS",
    "PARTNERSHIP",
    "PETROCHEMICAL",
    "PETROCHEMICALS",
    "PHARM",
    "PHARMA",
    "PHARMACEUTICA",
    "PHARMACEUTICAL",
    "PHARMACEUTICALS",
    "PHARAMACEUTICAL",
    "PHARAMACEUTICALS",
    "PLANT$",
    "PLC",
    "PROD",
    "PRODUCT",
    "PRODUCTS",
    "PTE",
    "PTY",
    "R&D",
    "R&Db",
    "REGENTS",
    "RES",
    "RES & TECH",
    "RES & DEV",
    "RESEARCH$",
    "SA",
    "SCI",
    "SCIENT",
    "SCIENCE",
    "SCIENCES",
    "SCIENTIFIC",
    "SE",
    "SERV",
    "SERVICE",
    "SERVICES",
    "SPA",
    "SYNTHELABO",
    "SYS",
    "SYST",
    "SYSTEM",
    "TECH",
    "TECHNOLOGY",
    "TECHNOLOGY SERVICES",
    "TECHNOLOGIES",
    "THERAPEUTIC",
]


COUNTRIES = [
    "CALISTOGA",
    "CANADA",
    "CHINA",
    "COLORADO",
    "DE",
    "DEUTSCHLAND",
    "EU",
    "NORTH AMERICA",
    "NA",
    "INDIA",
    "IRELAND",
    "JAPAN",
    "MA",
    "MFG",
    "PALO ALTO",
    "SAN DIEGO",
    "UK",
    "US",
    "USA",
]

COMPANY_SUPPRESSIONS = [
    *COMPANY_SUPPRESSIONS_DEFINITE,
    *COMPANY_SUPPRESSIONS_MAYBE,
    *[c for c in COUNTRIES],
]


COMPANY_MAP = {
    "massachusetts inst technology": "Massachusetts Institute of Technology",
    "roche": "Roche",
    "biogen": "Biogen",
    "boston scient": "Boston Scientific",
    "boston scimed": "Boston Scientific",
    "lilly co eli": "Eli Lilly",
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
    "koninkl philips": "Philips",
    "koninklijke philips": "Philips",
    "max planck": "Max Planck",
    "novartis": "Novartis",
    "pfizer": "Pfizer",
    "gilead": "Gilead",
    "dow": "Dow",
}

OWNER_TERM_MAP = {
    "lab": "laboratory",
    "labs": "laboratories",
    "univ": "university",
    "inst": "institute",
}

PATENT_ATTRIBUTE_MAP = {
    "COMBINATION": ["combo", "combination"],
    "COMPOUND_OR_MECHANISM": [
        *flatten(expand_res(ALL_INTERVENTION_BASE_TERMS)),
        "molecule",
        "receptor",
        "therapeutic",  # TODO: will probably over-match
    ],
    "DEVICE": [
        "apparatus",
        "cannula",
        "computer",
        "device",
        "drug matrix",
        "electronic",
        "implant",
        "injector",
        "instrument",
        "prosthesis",
        "scan",
        "sensor",
        "stent",
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
    ],
    "FORMULATION": [
        "formulation",
        "form",
        "salt",
        "extended release",
        "topical",
        "aerosol",
    ],
    "INCREMENTAL": ["improved process", "improved method", "improved system"],
    "IRRELEVANT": [
        "mammal",
        "veterinary",
        "animal",
        "nutritional",
        "food supplement",
        "primate",
        "traditional",
        "agricultural",
        "horticultural",
        "cosmetic",
        "fungicide",
        "herbicide",
        "pesticide",
        "plant disease",
        "pest control",
    ],
    "METHOD": ["procedure", "preparation method", "method of use"],
    "METHOD_OF_ADMINISTRATION": [
        "delivery",
        "dosing",
        "kit",
        "regimen",
        "administration",
    ],
    "NOVEL": ["novel"],
    "PALLIATIVE": [
        "palliative",
        "reduction in symptoms",
        "supportive",
        "symptom management",
        "symptom relief",
        "symptom relief",
    ],
    "PREVENTATIVE": ["prevention", "prophylaxis", "prophylactic"],
    "PROCEDURE": [
        "procedure",
        "surgery",
        "surgical",
    ],
    "PROCESS": [
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


PATENT_ATTRIBUTES = dict([(k, k) for k in PATENT_ATTRIBUTE_MAP.keys()])

SUITABILITY_SCORE_MAP: SuitabilityScoreMap = {
    "COMBINATION": -1,
    "COMPOUND_OR_MECHANISM": 3,
    "DEVICE": -2,
    "DIAGNOSTIC": -2,
    "DISEASE_MODIFYING": 1.5,
    "FORMULATION": -0.25,
    "IRRELEVANT": -5,
    "METHOD": -0.25,  # really only a prob if not COM
    "METHOD_OF_ADMINISTRATION": -1.0,
    "NOVEL": 1.5,
    "PALLIATIVE": 0,
    "PREVENTATIVE": 1,
    "PROCEDURE": -1.5,
    "PROCESS": -1,
    "TREATMENT": 0,
}


ATTRIBUTE_FIELD = "attributes"
