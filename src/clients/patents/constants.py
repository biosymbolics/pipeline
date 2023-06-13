"""
Constants for the patents client.
"""
from constants.patterns.moa import ACTIONS


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

# C07D217/04 - process for preparing
# C07C317/44 - method of use and making
# C07D405/12 - process for preparing
# C07C213/00 - transdermal delivery
# C07D405/12 - process for preparing
# C12Q1/68 - methods for characterization, monitoring, and treatment
# C12Q1/68 - method for predicting
# C12Q1/68 - pcr for detection
# C07D405/12, C07D405/14, C07D401/14 - process for preparing
# C07C215/08, C07C213/02 - process for preparing
# C07D333/70, C07D409/12 - methods of preparing
# C07D295/096 - process for preparing
# C07D311/26, C07C15/14, C07C15/12, C07C39/16, C07F15/00, C07F15/02, C07F15/04 - good
# C07D487/04, C07D471/04, C07D519/00, C07D471/04, C07D487/04, C07D519/00, C07D513/04 - good
# C07D209/40, C07D305/08, C07D295/135, C07D471/10, C07D487/10, C07D213/64 - good
# C07D413/04, C07D409/14, C07C317/44, C07J9/00, C07J7/00 - composition of matter AND methods
# C12Q1/68 - method for selecting
# C12Q1/68 - methods, processes, devices for measurement


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

COMPANY_NAME_SUPPRESSIONS = [
    "AB",
    "AG",
    "BIOSCIENCE",
    "BIOTEC",
    "BIOTECH",
    "CHEM",
    "CO",
    "COMPANY",
    "COOP",
    "CORP",
    "CORPORATION",
    "DEV",
    "DIAGNOSTIC",
    "FARM",
    "FARMA",
    "GROUP",
    "GMBH",
    "HEALTHCARE",
    "INC",
    "INSTITUTE",
    "INT",
    "IP",
    "KG",
    "LLC",
    "LP",
    "LTD",
    "MEDICAL",
    "NETWORK",
    "NV",
    "PATENT",
    "PHARM",
    "PHARMA",
    "PHARMACEUTICA",
    "PHARMACEUTICAL",
    "PHARAMACEUTICAL",
    "PTY",
    "SCIENCE",
    "SA",
    "SPA",
    "SYSTEM",
    "THE",
    "THERAPEUTIC",
]

COUNTRIES = [
    "DEUTSCHLAND",
    "EU",
    "IRELAND",
    "UK",
    "USA",
]

MAX_PATENT_LIFE = 20

PATENT_ATTRIBUTE_MAP = {
    "COMBINATION": ["combination"],
    "COMPOUND_OR_MECHANISM": [
        "composition",
        "compound",
        "composition",
        "derivative",
        "drug",
        "receptor",
        "substitute",
        *ACTIONS,
    ],
    "DIAGNOSTIC": [
        "analysis",
        "diagnosis",
        "diagnostic",
        "biomarker",
        "detection",
        "imaging",
        "marker",
        "monitoring",
        "risk score",
        "sensor",
        "testing",
    ],
    "FORMULATION": ["formulation", "form"],
    "METHOD": ["method", "procedure", "use"],
    "NOVEL": ["novel"],
    "PROCESS": ["preparation", "process", "sythesis", "system"],
}

PATENT_ATTRIBUTES = dict([(k, k) for k in PATENT_ATTRIBUTE_MAP.keys()])
