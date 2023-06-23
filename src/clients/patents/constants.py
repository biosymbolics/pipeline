"""
Constants for the patents client.
"""
from constants.patterns import MOA_ACTIONS

from .types import RelevancyThreshold


RELEVANCY_THRESHOLD_MAP: dict[RelevancyThreshold, float] = {
    "very low": 0.0,
    "low": 0.05,
    "medium": 0.20,
    "high": 0.50,
    "very high": 0.75,
}

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
    "^\\s*-",
    "^\\s*&",
]

COMPANY_SUPPRESSIONS_MAYBE = [
    "AGENCY",
    "AGROBIO",
    "AGROSCIENCE",
    "AGROSCIENCES",
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
    "INSTITUTE",
    "INST",
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
    "»",
    "«",
    "\\(.+\\)",  # remove any parantheticals
]

COUNTRIES = [
    "CALISTOGA",
    "CANADA",
    "COLORADO",
    "DE",
    "DEUTSCHLAND",
    "EU",
    "NORTH AMERICA",
    "NA",
    "IRELAND",
    "JAPAN",
    "PALO ALTO",
    "SAN DIEGO",
    "UK",
    "US",
    "USA",
]

COMPANY_SUPPRESSIONS = [
    *COMPANY_SUPPRESSIONS_DEFINITE,
    *COMPANY_SUPPRESSIONS_MAYBE,
    *COUNTRIES,
]

MAX_PATENT_LIFE = 20

PATENT_ATTRIBUTE_MAP = {
    "COMBINATION": ["combination"],
    "COMPOUND_OR_MECHANISM": [
        *MOA_ACTIONS,
        "composition",
        "compound",
        "composition",
        "derivative",
        "drug",
        "enzyme",
        "inhibiting",  # not the same stem as inhibitor
        "molecule",
        "receptor",
        "substitute",
        "therapy",  # TODO: will probably over-match
        "therapeutic",  # TODO: will probably over-match
    ],
    "DIAGNOSTIC": [
        "analysis",
        "diagnosis",
        "diagnostic",
        "diagnose",
        "biomarker",
        "detection",
        "imaging",
        "marker",
        "monitoring",
        "mouse",  # in vivo models
        "predict",
        "prognosis",
        "prognostic",  # needed?
        # "risk score",
        "scan",
        "sensor",
        "testing",
    ],
    "DISEASE_MODIFYING": [
        # "disease modifying",
        # "disease-modifying",
    ],
    "FORMULATION": ["formulation", "form", "salt"],
    "METHOD": ["method", "procedure"],  # preparation method, method of use
    "NOVEL": ["novel"],
    "PREVENTATIVE": ["prevention", "prophylaxis"],
    "PROCESS": [
        "preparation",
        "process",
        "synthesis",
        "system",
        "produce",
    ],  # method of making, method for producing
    "PALLIATIVE": [
        "palliative",
        # "reduction in symptoms",
        "supportive",
        # "symptom management",
        # "symptom relief",
        # "symptom relief"
    ],
}

PATENT_ATTRIBUTES = dict([(k, k) for k in PATENT_ATTRIBUTE_MAP.keys()])
