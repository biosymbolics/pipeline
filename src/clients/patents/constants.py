"""
Constants for the patents client.
"""
from constants.patterns import MOA_ACTIONS

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
        "biomarker",
        "detection",
        "imaging",
        "marker",
        "monitoring",
        "mouse",  # in vivo models
        "prognosis",
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
    "PREVENTATIVE": ["prevention"],
    "PROCESS": ["preparation", "process", "synthesis", "system"],
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
