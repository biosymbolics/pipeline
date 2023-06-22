"""
Patterns related to IUPAC nomenclature
"""

import re

from common.utils.re import get_or_re, ALPHA_CHARS


def compose_iupac_re() -> str:
    """
    Extracts IUPAC from string

    returns "(?i)\b((?:[\\wͰ-Ͽ()\\]\\[,-]*(?:alky|deoxy|tetr|...){1,100}[\\wͰ-Ͽ()\\]\\[,-]*){2,})(?:[ \n]|$)"
    """
    strings = [
        "alkyl",
        "deoxy",
        "ingenols",
        "tetr",
        "meth",
        # "prop",
        "pent",
        "hept",
        # "hect",
        "benz",
        # "phen",
        "acet",
        "oxol",
        "oxoc",
        "oxon",
        "iodo",
        "sila",
        "bora",
        "oate",
        # "urea",
        "furo",
        "oxaz",
        "oxan",
        "thia",
        "thio",
        "idin",
        "neon",
        "iron",
        "zinc",
        # "gold",
        # "lead",
        "hydro",
        "cyclo",
        "spiro",
        "oxido",
        "oxino",
        "oxalo",
        "bromo",
        "stiba",
        "sulfo",
        "nitro",
        "amino",
        "amido",
        "imino",
        "imido",
        "cyano",
        "azido",
        "diazo",
        "amine",
        "imine",
        "amide",
        "imide",
        "carbo",
        "silyl",
        "boron",
        "oxide",
        "inden",
        "cumen",
        "picen",
        "pyren",
        "cuban",
        "indol",
        "furan",
        "purin",
        "pyran",
        "borol",
        "selen",
        "idene",
        "triaz",
        "argon",
        "xenon",
        "aurum",
        "radon",
        "peroxy",
        "pyrido",
        "protio",
        "tritio",
        "fluoro",
        "chloro",
        "iodido",
        "iodide",
        "iodane",
        "arsono",
        "germyl",
        "azanyl",
        "azonia",
        "azonio",
        "cyanic",
        "formic",
        "formyl",
        "silole",
        "peroxo",
        "azulen",
        "ovalen",
        "pyrrol",
        "acrido",
        "borino",
        "corrin",
        "phosph",
        "tellur",
        "trityl",
        "helium",
        "oxygen",
        "sodium",
        "sulfur",
        "cobalt",
        "nickel",
        "copper",
        "silver",
        "indium",
        "iodine",
        "cesium",
        "barium",
        "cerium",
        "erbium",
        "osmium",
        "radium",
        "curium",
        "hydrate",
        "imidazo",
        "naphtho",
        "stibolo",
        "protide",
        "bromido",
        "iodanyl",
        "bromide",
        "arsanyl",
        "arsonic",
        "stannyl",
        "plumbyl",
        "gallane",
        "germane",
        "sulfino",
        "sulfido",
        "sulfite",
        "sulfate",
        "sulfide",
        "sulfane",
        "cyanato",
        "azanide",
        "azanida",
        "amidino",
        "oxamide",
        "amidine",
        "azanium",
        "cyanate",
        "cyanide",
        "nitrile",
        "nitrite",
        "formate",
        "carbamo",
        "borinic",
        "selanyl",
        "adamant",
        "fluoren",
        "chrysen",
        "coronen",
        "perylen",
        "pyridin",
        "pyrazol",
        "chromen",
        "xanthen",
        "pyrazin",
        "indazol",
        "acridin",
        "borinin",
        "boriran",
        "borinan",
        "indacen",
        "silonin",
        "borepin",
        "lithium",
        "silicon",
        "calcium",
        "gallium",
        "arsenic",
        "bromine",
        "krypton",
        "yttrium",
        "niobium",
        "rhodium",
        "cadmium",
        "latinum",
        "terbium",
        "holmium",
        "thulium",
        "hafnium",
        "rhenium",
        "iridium",
        "mercury",
        "bismuth",
        "thorium",
        "uranium",
        "fermium",
        "dubnium",
        "bohrium",
        "hassium",
        "nitrate",
        "hydride",
        "isobutyl",
        "pyrimido",
        "stiboryl",
        "deuterio",
        "fluorido",
        "chlorido",
        "bromanyl",
        "fluoride",
        "chloride",
        "gallanyl",
        "chromium",
        "stannane",
        "tellanyl",
        "germanyl",
        "sulfanyl",
        "sulfinyl",
        "sulfamic",
        "sulfinic",
        "azanidyl",
        "ammonium",
        "hydrazin",
        "carbamic",
        "phthalic",
        "naphthal",
        "carbanid",
        "fulleren",
        "pleiaden",
        "arsindol",
        "cinnolin",
        "carbazol",
        "guanidin",
        "yohimban",
        "pteridin",
        "quinolin",
        "triamine",
        "fluorine",
        "chlorine",
        "scandium",
        "titanium",
        "vanadium",
        "rubidium",
        "samarium",
        "europium",
        "lutetium",
        "tantalum",
        "tungsten",
        "platinum",
        "thallium",
        "polonium",
        "astatine",
        "francium",
        "actinium",
        "nobelium",
        "silicate",
        "isopropyl",
        "sec-butyl",
        "fluoranyl",
        "chloranyl",
        "chloridic",
        "sulfamoyl",
        "sulfinato",
        "sulfenato",
        "sulfinate",
        "sulfanium",
        "sulfamate",
        "hydrazono",
        "formamido",
        "nitramido",
        "hydrazide",
        "nitramide",
        "formamide",
        "cyanamide",
        "carbamate",
        "phthalate",
        "piperazin",
        "pyrimidin",
        "piperidin",
        "morpholin",
        "pyridazin",
        "porphyrin",
        "perimidin",
        "trisodium",
        "triacetyl",
        "beryllium",
        "magnesium",
        "potassium",
        "manganese",
        "germanium",
        "zirconium",
        "ruthenium",
        "palladium",
        "lanthanum",
        "neodymium",
        "ytterbium",
        "neptunium",
        "plutonium",
        "americium",
        "berkelium",
        "periodate",
        "tert-butyl",
        "iodanuidyl",
        "perchloric",
        "hypoiodous",
        "isocyanide",
        "carbamimid",
        "quinazolin",
        "phthalazin",
        "quinoxalin",
        "quinolizin",
        "molybdenum",
        "technetium",
        "promethium",
        "gadolinium",
        "dysprosium",
        "lawrencium",
        "seaborgium",
        "meitnerium",
        "perbromate",
        "hypoiodite",
        "sulfinamoyl",
        "sulfinamide",
        "cyanatidoyl",
        "hydrazonate",
        "formonitril",
        "fluoranthen",
        "einsteinium",
        "californium",
        "mendelevium",
        "perchlorate",
        "hypobromite",
        "hypochlorous",
        "carbaldehyde",
        "halocarbonyl",
        "terephthalic",
        "formaldehyde",
        "aceanthrylen",
        "naphthyridin",
        "praseodymium",
        "protactinium",
        "hypofluorite",
        "hypochlorite",
        "sulfinimidoyl",
        "terephthalate",
        "acenaphthylen",
        "rutherfordium",
        "phthalaldehyde",
        "acephenanthrylen",
        "terephthalaldehyde",
    ]
    extra_chars = [r"(", r")", r"\]", r"\[", ",", r"-"]
    iupac_substr = ALPHA_CHARS("*", None, extra_chars)
    regex = (
        r"\b((?:"
        + iupac_substr
        + get_or_re(strings, 1)
        + iupac_substr
        + r"){1,})(?:[ \n]|$)"
    )
    return regex


IUPAC_RE = compose_iupac_re()


def extract_iupac(string: str) -> list[str]:
    """
    Extracts IUPAC from string

    Usage:
        >>> extract_iupac("1-(3-aminophenyl)-6,8-dimethyl-5-(4-iodo-2-fluoro-phenylamino)-3-cyclopropyl-1h,6h-pyrido[4,3-d]pyridine-2,4,7-trione derivatives as mek 1/2 inhibitors")
        >>> extract_iupac("3-alkylamido-3-deoxy-ingenols")
    """
    return [match.strip() for match in re.findall(IUPAC_RE, string)]