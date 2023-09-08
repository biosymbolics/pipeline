"""
To run after NER is complete
"""
import sys
import logging
from typing import Literal
from pydash import compact, flatten

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient as DatabaseClient
from constants.core import (
    SOURCE_BIOSYM_ANNOTATIONS_TABLE as SOURCE_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE as WORKING_TABLE,
)
from constants.patterns.intervention import (
    COMPOUND_BASE_TERMS_GENERIC,
    MECHANISM_BASE_TERMS,
    INTERVENTION_BASE_TERMS,
    INTERVENTION_PREFIXES,
)
from utils.re import get_or_re


TextField = Literal["title", "abstract"]
WordPlace = Literal["leading", "trailing", "all"]


TEXT_FIELDS: list[TextField] = ["title", "abstract"]
REMOVAL_WORDS_PRE: dict[str, WordPlace] = {
    "such": "all",
    "method": "all",
    "obtainable": "all",
    "different": "all",
    "stable": "all",
    "various": "all",
    "the": "leading",
    "example": "all",
    "unwanted": "leading",
    "comprised?": "all",
    "contagious": "leading",
    "recognition": "trailing",
    "binding": "trailing",
    "prevention": "leading",
    "encoding": "trailing",
    "discreet": "all",
    "properties": "trailing",
    "derived": "all",
    "library": "all",
    "more": "leading",
    "classic": "all",
    "excellent": "all",
    "construct": "trailing",
    "particular": "all",
    "useful": "all",
    "uses(?: thereof| of)": "all",
    "designer": "all",
    "thereof": "all",
    "capable": "all",
    "specific": "all",
    "in": "leading",
    "recombinant": "all",
    "novel": "all",
    "exceptional": "all",
    "non[ -]?toxic": "leading",
    "(?:non )?selective": "leading",
    "adequate": "leading",
    "improved": "all",
    "improving": "all",
    "new": "leading",
    r"\y[(]?e[.]?g[.]?,?": "all",
    "-targeted": "all",
    "long[ -]?acting": "leading",
    "potent": "all",
    "inventive": "all",
    "other": "leading",
    "more": "leading",
    "of": "trailing",
    "symptom": "trailing",
    "condition": "trailing",
    "be": "trailing",
    "use": "trailing",
    "efficacy": "all",
    "advanced": "all",
    "promising": "all",
    "therapeutic procedure": "all",
    "therapeautic?": "all",
    "therapeutic(?:ally)?": "all",
    "therefor": "all",
    "prophylactic": "all",
    "(?:co[ -]?)?therapy": "trailing",
    "(?:pharmaceutical |chemical )?composition": "trailing",
    "treatment method": "all",
    "treatment with": "all",
    "treating": "all",
    "contact": "trailing",
    "portion": "trailing",
    "intermediate": "all",
    "suitable": "all",
    "procedure": "all",  # TODO
    "relevant": "all",
    "patient": "all",
    "acceptable": "all",
    "thereto": "all",
    "exemplary": "all",
    "against": "trailing",
    "usable": "all",
    "other": "leading",
    "suitable": "all",
    "use of": "leading",
    "certain": "all",
    "working": "leading",
    "on": "trailing",
    "in(?: a)?": "trailing",
    "(?: ,)?and": "trailing",
    "and ": "leading",
    "the": "trailing",
    "with": "trailing",
    "of": "trailing",
    "for": "trailing",
    "=": "trailing",
    "unit(?:[(]s[)])?": "trailing",
    "measurement": "all",
    "measuring": "all",
    "system": "trailing",
    "[.]": "trailing",
    "analysis": "all",
    "management": "all",
    "accelerated": "all",
    "below": "trailing",
    "fixed": "leading",
    "pharmacological": "all",
    "acquisition": "all",
    "production": "all",
    "level": "trailing",
    "processing(?: of)?": "all",
    "lead candidate": "all",
    "control": "trailing",
    "modified": "all",
    "variant": "trailing",
    "variety": "trailing",
    "varieties": "trailing",
    "family": "trailing",
    "(?:pharmaceutically|physiologically) (?:acceptable |active )?": "leading",
    "pure": "all",
    "chemically (?:modified)?": "all",
    "based": "trailing",
    "an?": "leading",
    "active": "all",
    "wherein": "all",
    "additional": "all",
    "additive": "all",
    "advantageous": "all",
    "aforementioned": "all",
    "aforesaid": "all",
    "efficient": "all",
    "first": "all",
    "second": "all",
    "abnormal": "all",
    "atypical": "all",
    "inappropriate": "all",
    "compounds as": "all",
    "formula [(][ivxab]{1,3}[)]": "trailing",
    "is": "leading",
    "engineered": "all",
    "medicament": "all",
    "medicinal": "all",
    "sufficient": "all",
    "due": "trailing",
    "locate": "all",
    "specification": "all",
    "detect": "all",
    "similar": "all",
    "contemplated": "all",
    "predictable": "all",
    "convenient": "all",
    "dosing": "leading",
    "preferred": "leading",
    "conventional": "leading",
    "clinically[ -]?proven": "all",
    "proven": "all",
    "contemplated": "all",
    "is indicative of": "all",
    "via": "leading",
    "effective": "all",
    "(?:high|low)[ -]?dose": "all",
    "effects of": "all",
    "soluble": "leading",
    "competitive": "leading",
    "mutant": "leading",
    "mutated": "leading",
    "activatable": "all",
    # model/source
    "murine": "all",
    "mouse": "all",
    "mice": "all",
    "human(?:ized|ised)?": "all",  # ??
    "rat": "all",
    "rodent": "all",
    "rabbit": "all",
    "porcine": "all",
    "bovine": "all",
    "equine": "all",
    "mammal(?:ian)?": "all",
}

REMOVAL_WORDS_POST: dict[str, WordPlace] = dict(
    [
        (t, "trailing")
        for t in [
            *COMPOUND_BASE_TERMS_GENERIC,
            "activity",
            "agent",
            "effect",
            "pro[ -]?drug",
            "mediated?",
        ]
    ]
)


DELETION_TERMS = [
    "crystal structure",
    "standard tibetan language",
    "present tricyclic",
    "topical surface",
    "cell",
    "wherein(?: a)?",
    "computer-readable",
    "pharmaceutical composition",
    "compound i",
    "wherein said compound",
    "fragment of",
    "pharmaceutically[- ]acceptable",
    "pharmacological composition",
    "pharmacological",
    "therapeutically",
    "general formula ([A-Z0-9]+)",
    "medicine compound",
    "receptacle",
    "liver",
    "unsubstituted",
    "compound i",
    "medicinal composition",
    "compound",
    "disease",
    "medicine compounds of formula",
    "therapy",
    "geographic location",
    "quantitation",
    "(?:administrative )?procedure",
    "endoscope",
    "optionally substituted",
    "suction",
    "compound 1",
    "prognosticating",
    "formula",
    "absorbent article",
    ".*plunger",
    "adhesive",
    "topical",
    "susceptible",
    "aqueous",
    "medicinal",
    "conduit",
    "pharmaceutically",
    "topical",
    "heterocyclic",
    "recombinant technique",
    "bioactive",
    "adhesive layer",
    "cellulose",
    "resin",
    "starch",
    "absorbent",
    "excipient",
    "amide",
    "amine",
    "single chain",
    "recombinantly",
    "aperture",
    "scaffold",
    "lipid",
    "intraocular len",
    "curable",
    "injectable",
    "absorbent body",
    "antibody[ -]?drug",
    "inorganic",
    "transdermal delivery",
    "nutraceutical",
    "solvent",
    "gastrointestinal tract",
    "sustained release",
    "target",
    "compartment",
    "electrically conductive",
    "cartilage",
    "therapeutically active",
    "sheath",
    "ablation",
    "conjugated",
    "polymeric",
    "ester",
    "ophthalmic",
    "silicone",
    "aromatic",
    "propylene",
    "biocompatible",
    "recombinant",
    "composition comprising",
    "computer-readable medium",
    "single nucleotide polymorphism",
    "popical",
    "transgene",
    "heterocyclic",
    "target protein",
    "biologically active",
    "polymerizable",
    "biosensor",
    "ophthalmic",
    "urea",
    "peptide",
    "bioactive agent",
    "protein expression",
    "lignin",
    "thermoplastic",
    "acetone",
    "sunscreen",
    "non[ -]?invasive",
    "glycerin",
    "medicinal",
    "cell culture",
    "indole",
    "benzene",
    "spectrometer",
    "aberrant expression",
    "crystallization",
    "oxidant",
    "phenolic",
    "hydrocarbon",
    "titanium dioxide",
    "antigen presenting cell",
    "amide",
    "taxane",
    "diamine",
    "gelatin",
    "ketone",
    # device
    "(?:.* )?hinge",
    "(?:.* )?dispenser",
    "(?:.* )?probe",
    "(?:.* )?rod",
    "(?:.* )?prosthesis",
    "(?:.* )?catheter.*",
    "(?:.* )?electrode",
    "(?:.* )?fastener",
    "(?:.* )?waveguide",
    "(?:.* )?spacer",
    "electromagnetic radiation",
    "(?:.* )?implant",
    "(?:.* )?actuator",
    "(?:.* )?clamp",
    "(?:.* )?mass spectrometer",
    "magnetic resonance imaging.*",
    "(?:.* )?fabric",
    "(?:.* )?diaper",
    "(?:.* )?coil",
    "(?:.* )?apparatus(?:es)?",
    "(?:.* )?sensor",
    "(?:.* )?wafer",
    "tampon",
    "absorbent pad",
    "(?:.* )?syringe(?: .*)?",
    "(?:.* )?canister",
    "bioreactor",
    "(?:.* )?tether",
    "(?:.* )?mouthpiece",
    "(?:.* )?transducer",
    "electrical stimulation",
    "(?:.* )?toothbrush",
    "(?:.* )?strut",
    "(?:.* )?suture",
    "(?:.* )?cannula",
    "(?:.* )?stent",
    "(?:.* )?capacitor",
    "(?:.* )?mass spectrometry",
    "accelerometer",
    "container",
    "reservoir",
    "elongated housing",
    "modulator device",
    "injector",
    "(?:.* )?diaphragm",
    "(?:.* )?cartridge",
    "(?:.* )?plunger",
    "ultrasound(?: .*)?",
    "(?:.* )?piston",
    "(?:.* )?microprocessor",
    "(?:.* )?balloon",
    "(?:.* )?stapler",
    "internal combustion engine",
    "capsule",
    "valve",
    "solubility",
    "compressor",
    "forcep",
    "beam splitter",
    ".*transceiver",
    ".*piezoelectric.*",
    ".*ultrasonic.*",
    "impeller",
    "(?:.* )?appliance",
    "transmitter",
    "tubing",
    "(?:.* )?tube",
    "(?:.* )?lancet",
    "slider",
    "(?:.* )?tomography" "(?:.* )?instrument",
    "abutment",
    "gasket",
    "(?:.* )?wave",
    "(?:.* )?pump",
    "(?:.* )?article",
    "(?:.* )?screw",
    "(?:.* )?cytometer",
    "interferometer",
    "inflatable bladder",
    "blower",
    "fastening mean",
    "piezoelectric.*",
    "handpiece",
    "reagent kit",
    "(?:.* )?diode",
    "anvil",
    "(?:.* )?chromatography",
    "(?:.* )?blade",
    "centrifugal force",
    "(?:.* )?needle",
    "implant .*",
    "tensile strength",
    "(?:.* )?wheelchair",
    "(?:.* )?machine",
    "(?:.* )?applicator",
    "(?:.* )?monitor",
    "(?:.* )?irrigator",
    "(?:.* )?accelerator",
    "(?:.* )?indicator",
    "(?:.* )?pump",
    "robot.*" "(?:.* )?sponge",
    # "(?:.* )?generator",??
    # end device
    "diluent",
    "bifunctional",
    "inhibitory",
    "tubular member",
    "specific antibodie",
    "catalytic",
    "gene delivery",
    "said protein",
    "fibrous",
    "in vitro",
    "polypeptides present",
    "volatile",
    "amino acid",
    "human(?:ized)? antibody",
    "human(?:ized)? antibodie",
    "silica",
    # procedure
    "(?:.* )?ablation",
    "(?:.* )?surger(?:y|ie)",
    "radiotherapy",
    "sealant",
    "microelectronic",
    # agtech
    "herbicide",
    "insecticide" "fungicide",
    "pesticide",
    "transgenic plant",
    "drought tolerance",
    "biofuel",
    "biodiesel",
    "plant growth regulator",
    ".* plant"
    # end agtech
    "explosive",
    "cell culture techniques",
    # diagnostic
    "(?:.* )?testing",
    "(?:.* )?detection",
    "(?:.* )?diagnostic",
    "(?:.* )?diagnosis",
    "analyte",
    "(?:.* )?scopy" "(?:.* )?reagent",
    "dielectric",
    "aroma",
    "crystalline",
    "edible",
    "saline",
    "pharmaceutically salt",
    "citrate",
    "non-therapeutic",
    "functional",
    "medicinal agent",
    "sucrose",
    "intramedullary nail",
    "liquid",
    "unsaturated",
    "adhesion",
    "tibial",
    ".* atrium",
    "topsheet",
    "biologically active agent",
    "pharmaceutically active",
    "therapeutically-effective amount",
    "cross[ -]?linking",
    "biocompatibility",
    "porous",
    "intraocular lens",
    "dispensing",
    "impedance",
    "radioactive",
    "prevention of cancer",
    "endotracheal tube",
    "cancer diagnosis",
    "biologically active agent",
    "pesticidal activity",
    r"compound\(s\)",
    "therapeutical",
    "ingredient",
    "conductive",
    "elastic",
    "microcapsule",
    "hydrophilic",
    "(?:medication |drug |therapy |dose |dosing |dosage |treatment |therapeutic |administration |daily |multiple |delivery |suitable |chronic |suitable |clinical |extended |convenient |effective |detailed |present )+regimen",
    "stylet",
    "monotherapy",
    "aerosol",
    "pharmacologically active agent",
    "left atrium",
    "sulfur",
    "quantification",
    "computer-implemented",
    "flexible",
    "corresponding",
    "alkylation",
    "mandrel",
    "macrocyclic",
    "pharmaceutically active agent",
    "polymer",
    "agent",
    "absorbent core",
    "heart valve",
    "nonwoven",
    "curable",
    "sanitary napkin",
    ".*catheter.*",
    "extracellular vesicle",
    "target antigen",
    "water-soluble",
    "illustrative",
    "metal",
    "superabsorbent",
    "expandable member",
    "lipase",
    "femoral",
    "obturator",
    "fructose",
    "respiratory",
    "said antibody",
    "computer[ -]?readable",
    "sweetener",
    ".*administration",
    ".*patient",
    "field of .*",
    "femur",
    "immunogenic",
    "organic solvent",
    "bacterium",
    "bacteria",
    "sterol",
    "nucleic acid sequencing",
    "ethylene",
    "keratin fib(?:re|er)s?",
    "dermatological",
    "tubular body",
    "protease",
    "antigen-binding",
    "pyridine",
    "pyrimidine",
    "phenol",
    "said",
    "reporter",
    "solvate",
    "nutrient",
    "sterilization",
    "carbonyl",
    "aldehyde",
    "cancer stem cell",
    "cancer cell",
    "cross[- ]?linked",
    "nucleic acid",
    "elongated body",
    "lactic acid",
    "oligomeric",
    "(?:.* )? delivery",
    "ammonia",  # comp
    "keratin",
    "trocar",
    "enzymatic",
    "volatile organic",
    "fluorescent",
    "regeneration",
    "emulsion",
    "resilient",
    "biodegradable",
    "biomaterial",
    "T cell receptor",
    "cleansing",
    "lipophilic",
    "propane",
    "elongated shaft",
    "transdermal",
    "brachytherapy",
    "particulate",
    ".* delivery",
    "perfume",
    "cosmetic",
]


def remove_substrings():
    """
    Removes substrings from annotations
    (annotations that are substrings of other annotations for that publication_number)
    """
    temp_table = "names_to_remove"
    query = rf"""
        SELECT t1.publication_number AS publication_number, t2.term AS removal_term
        FROM {WORKING_TABLE} t1
        JOIN {WORKING_TABLE} t2
        ON t1.publication_number = t2.publication_number
        WHERE t2.term<>t1.term
        AND t1.term ~* CONCAT('.*', escape_regex_chars(t2.term), '.*')
        AND length(t1.term) > length(t2.term)
        AND array_length(regexp_split_to_array(t2.term, '\s+'), 1) < 3
        ORDER BY length(t2.term) DESC
    """

    delete_query = f"""
        DELETE FROM {WORKING_TABLE}
        WHERE ARRAY[publication_number, term] IN (
            SELECT ARRAY[publication_number, removal_term]
            FROM {temp_table}
        )
    """

    logging.info("Removing substrings")
    client = DatabaseClient()

    client.create_from_select(query, temp_table)
    client.execute_query(delete_query)
    client.delete_table(temp_table)


def fix_of_for_annotations():
    """
    Handles "inhibitors of XYZ" and the like, which neither GPT or SpaCyNER were good at finding
    (but high hopes for binder)
    """
    logging.info("Fixing of/for annotations")

    terms = INTERVENTION_BASE_TERMS
    prefixes = INTERVENTION_PREFIXES

    prefix_re = "|".join([p + " " for p in prefixes])

    # inhibition of apoptosis signal-regulating kinase 1 (ASK1)
    def get_query(re_term: str, field: TextField):
        sql = f"""
            UPDATE {WORKING_TABLE} ba
            SET term=(substring({field}, '(?i)((?:{prefix_re})*{re_term} (?:of |for |the |that |to |comprising |(?:directed |effective |with efficacy )?against )+(?:(?:the|a) )?.*?)(?:and|useful|for|,|.|$)'))
            FROM applications a
            WHERE ba.publication_number=a.publication_number
            AND term ~* '^(?:{prefix_re})*{re_term}$'
            AND a.{field} ~* '.*{re_term} (?:of|for|the|that|to|comprising|against|(?:directed |effective |with efficacy )?against).*'
        """
        return sql

    def get_hyphen_query(term, field: TextField):
        re_term = term + "s?"
        sql = f"""
            UPDATE {WORKING_TABLE} ba
            SET term=(substring(title, '(?i)([A-Za-z0-9]+-{re_term})'))
            FROM applications a
            WHERE ba.publication_number=a.publication_number
            AND term ~* '^{re_term}$'
            AND a.{field} ~* '.*[A-Za-z0-9]+-{re_term}.*'
        """
        return sql

    client = DatabaseClient()

    for term in terms:
        for field in TEXT_FIELDS:
            try:
                sql = get_hyphen_query(term, field)
                client.execute_query(sql)
            except Exception as e:
                logging.error(e)

    # loop over term sets, in which the term may be in another form than the title variant
    for term in terms:
        for field in TEXT_FIELDS:
            try:
                sql = get_query(term, field)
                client.execute_query(sql)
            except Exception as e:
                logging.error(e)


def remove_trailing_leading(removal_word_set: dict[str, WordPlace]):
    logging.info("Removing trailing/leading words")

    # \y === \b in postgres re
    def get_remove_words():
        def get_sql(place: str):
            if place == "trailing":
                words = [
                    t[0] + "s?[ ]*" for t in removal_word_set.items() if t[1] == place
                ]
                if len(words) == 0:
                    return None
                words_re = get_or_re(words, "+")
                return rf"""
                    update {WORKING_TABLE} set term=(REGEXP_REPLACE(term, '\y{words_re}$', '', 'gi'))
                    where term ~* '.*\y{words_re}$'
                """
            elif place == "leading":
                words = [
                    t[0] + r"s?[ ]*" for t in removal_word_set.items() if t[1] == place
                ]
                if len(words) == 0:
                    return None
                words_re = get_or_re(words, "+")
                return rf"""
                    update {WORKING_TABLE} set term=(REGEXP_REPLACE(term, '^{words_re}\y', '', 'gi'))
                    where term ~* '^{words_re}\y.*'
                """
            elif place == "all":
                words = [
                    t[0] + "s?[ ]*" for t in removal_word_set.items() if t[1] == place
                ]
                if len(words) == 0:
                    return None
                words_re = get_or_re(words, "+")
                return rf"""
                    update {WORKING_TABLE} set term=(REGEXP_REPLACE(term, '\y{words_re}\y', ' ', 'gi'))
                    where term ~* '.*\y{words_re}\y.*'
                """
            else:
                raise ValueError(f"Unknown place: {place}")

        return compact([get_sql(place) for place in ["all", "leading", "trailing"]])

    client = DatabaseClient()
    for sql in get_remove_words():
        client.execute_query(sql)


def clean_up_junk():
    """
    Remove trailing junk and silly matches
    """
    logging.info("Removing junk")

    queries = [
        # unwrap
        f"update {WORKING_TABLE} "
        + r"set term=(REGEXP_REPLACE(term, '[)(]', '', 'g')) where term ~ '^[(][^)(]+[)]$'",
        rf"update {WORKING_TABLE} set term=(REGEXP_REPLACE(term, '^\"', '')) where term ~ '^\"'",
        # orphaned closing parens
        f"update {WORKING_TABLE} set term=(REGEXP_REPLACE(term, '[)]', '')) where term ~ '.*[)]' and not term ~ '.*[(].*';",
        # leading/trailing whitespace
        rf"update {WORKING_TABLE} set term=trim(term) where trim(term) <> term",
        # fixes some sentences
        f"update {WORKING_TABLE} set "
        + "term=regexp_extract(term, '(.{10,})(?:[.] [A-Z][A-Za-z0-9]{3,}).*') where term ~ '.{10,}[.] [A-Z][A-Za-z0-9]{3,}'",
    ]
    client = DatabaseClient()
    for sql in queries:
        client.execute_query(sql)


def fix_unmatched():
    """
    Example: 3 -d]pyrimidine derivatives -> Pyrrolo [2, 3 -d]pyrimidine derivatives
    """

    logging.info("Fixing unmatched parens")

    def get_query(field, char_set):
        sql = f"""
            UPDATE {WORKING_TABLE} ab
            set term=substring(a.{field}, CONCAT('(?i)([^ ]*{char_set[0]}.*', escape_regex_chars(term), ')'))
            from applications a
            WHERE ab.publication_number=a.publication_number
            AND substring(a.{field}, CONCAT('(?i)([^ ]*{char_set[0]}.*', escape_regex_chars(term), ')')) is not null
            AND term ~* '.*{char_set[1]}.*' AND not term ~* '.*{char_set[0]}.*'
            AND {field} ~* '.*{char_set[0]}.*{char_set[1]}.*'
        """
        return sql

    client = DatabaseClient()
    for field in TEXT_FIELDS:
        for char_set in [(r"\[", r"\]"), (r"\(", r"\)")]:
            sql = get_query(field, char_set)
            client.execute_query(sql)


def remove_common_terms():
    """
    Remove common original terms
    """
    logging.info("Removing common terms")
    # regex in here, effectively ignored
    common_terms = [
        *DELETION_TERMS,
        *flatten(INTERVENTION_BASE_TERMS),
        *INTERVENTION_BASE_TERMS,
    ]

    or_re = get_or_re([f"{t}s?" for t in common_terms])
    common_del_query = f"""
        delete from {WORKING_TABLE}
        where
        term=''
        OR term is null
        OR term ~* '^{or_re}$'
    """

    del_queries = [
        common_del_query,
        f"delete FROM {WORKING_TABLE} "
        + r"where term ~* '^[(][0-9a-z]{1,4}[)]?[.,]?[ ]?$'",
        f"delete FROM {WORKING_TABLE} "
        + r"where term ~ '^[0-9., ]+$'",  # only numbers . and ,
        rf"delete from {WORKING_TABLE} where length(term) > 150 and term ~* '\y(?:and|or)\y';",  # sentences
        rf"delete from {WORKING_TABLE}  where length(term) > 150 and term ~* '.*[.;] .*';",  # sentences
        f"delete FROM {WORKING_TABLE} where term ~* '^said .*'",  # arg
        f"delete FROM {WORKING_TABLE} where length(trim(term)) < 3 or term is null",
    ]
    for del_query in del_queries:
        DatabaseClient().execute_query(del_query)


def normalize_domains():
    """
    Normalizes domains
        - by rules
        - if the same term is used for multiple domains, pick the most common one
    """
    client = DatabaseClient()

    mechanism_terms = [f"{t}s?" for t in MECHANISM_BASE_TERMS]
    mechanism_re = get_or_re(mechanism_terms)

    queries = [
        f"update {WORKING_TABLE} set domain='mechanisms' where domain<>'mechanisms' AND term ~* '.*{mechanism_re}$'",
        f"update {WORKING_TABLE} set domain='mechanisms' where domain<>'mechanisms' AND term in ('abrasive', 'dyeing', 'dialyzer', 'colorant', 'herbicidal', 'fungicidal', 'deodorant', 'chemotherapeutic',  'photodynamic', 'anticancer', 'anti-cancer', 'tumor infiltrating lymphocytes', 'electroporation', 'vibration', 'disinfecting', 'disinfection', 'gene editing', 'ultrafiltration', 'cytotoxic', 'amphiphilic', 'transfection', 'chemotherapy')",
        f"update {WORKING_TABLE} set domain='diseases' where term in ('adrenoleukodystrophy', 'stents') or term ~ '.* diseases?$'",
        f"update {WORKING_TABLE} set domain='compounds' where term in ('ethanol', 'isocyanates')",
        f"update {WORKING_TABLE} set domain='compounds' where term ~* '(?:^| |,)(?:molecules?|molecules? bindings?|reagents?|derivatives?|compositions?|compounds?|formulations?|stereoisomers?|analogs?|analogues?|homologues?|drugs?|regimens?|clones?|particles?|nanoparticles?|microparticles?)$' and not term ~* '(anti|receptor|degrade|disease|syndrome|condition)' and domain<>'compounds'",
        f"update {WORKING_TABLE} set domain='mechanisms' where term ~* '.*receptor$' and domain='compounds'",
        f"update {WORKING_TABLE} set domain='diseases' where term ~* '(?:cancer|disease|disorder|syndrome|autism|condition|psoriasis|carcinoma|obesity|hypertension|neurofibromatosis|tumor|tumour|glaucoma|arthritis|seizure|bald|leukemia|huntington|osteo|melanoma|schizophrenia)s?$' and not term ~* '(?:treat(?:ing|ment|s)?|alleviat|anti|inhibit|modul|target|therapy|diagnos)' and domain<>'diseases'",
        f"update {WORKING_TABLE} set domain='mechanisms' where term ~* '.*gene$' and domain='diseases' and not term ~* '(?:cancer|disease|disorder|syndrome|autism|associated|condition|psoriasis|carcinoma|obesity|hypertension|neurofibromatosis|tumor|tumour|glaucoma|retardation|arthritis|tosis|motor|seizure|bald|leukemia|huntington|osteo|atop|melanoma|schizophrenia|susceptibility|toma)'",
        f"update {WORKING_TABLE} set domain='mechanisms' where term ~* '.* factor$' and not term ~* '.*(?:risk|disease).*' and domain='diseases'",
        f"update {WORKING_TABLE} set domain='mechanisms' where term ~* 'receptors?$' and domain='diseases'",
    ]
    for sql in queries:
        client.execute_query(sql)

    normalize_sql = f"""
        WITH ranked_domains AS (
            SELECT
                lower(term) as lot,
                domain,
                ROW_NUMBER() OVER (PARTITION BY lower(term) ORDER BY COUNT(*) DESC) as rank
            FROM {WORKING_TABLE}
            GROUP BY lower(term), domain
        )
        , max_domain AS (
            SELECT
                lot,
                domain AS new_domain
            FROM ranked_domains
            WHERE rank = 1
        )
        UPDATE {WORKING_TABLE} ut
        SET domain = md.new_domain
        FROM max_domain md
        WHERE lower(ut.term) = md.lot and ut.domain <> md.new_domain;
    """

    # update is much faster w/o this index, and it isn't needed from here on out anyway
    client.execute_query("drop index trgm_index_biosym_annotations_term")
    client.execute_query(normalize_sql)


def populate_working_biosym_annotations():
    """
    - Copies biosym annotations from source table
    - Performs various cleanups and deletions
    """
    client = DatabaseClient()
    logging.info(
        "Copying source (%s) to working (%s) table", SOURCE_TABLE, WORKING_TABLE
    )
    client.create_from_select(
        f"SELECT * from {SOURCE_TABLE} where domain<>'attributes'",
        WORKING_TABLE,
    )

    # add indices after initial load
    client.create_indices(
        [
            {"table": WORKING_TABLE, "column": "publication_number"},
            {"table": WORKING_TABLE, "column": "term", "is_tgrm": True},
        ]
    )

    fix_unmatched()
    clean_up_junk()

    # round 1 (leaves in stuff used by for/of)
    remove_trailing_leading(REMOVAL_WORDS_PRE)
    fix_of_for_annotations()

    # round 2 (removes trailing "compound" etc)
    remove_trailing_leading(REMOVAL_WORDS_POST)

    # clean up junk again (e.g. leading ws)
    # check: select * from biosym_annotations where term ~* '^[ ].*[ ]$';
    # select term from biosym_annotations where length(term) > 150 and term like '%and%';
    clean_up_junk()

    remove_common_terms()  # remove one-off generic terms
    remove_substrings()  # less specific terms in set with more specific terms

    # normalize_domains is **much** faster w/o this index
    normalize_domains()

    # do this last to minimize mucking with attribute annotations
    client.select_insert_into_table(
        f"SELECT * from {SOURCE_TABLE} where domain='attributes'", WORKING_TABLE
    )


if __name__ == "__main__":
    """
    Checks:

    select sum(count) from (select count(*) as count from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and term<>'' group by lower(term) order by count(*) desc limit 1000) s;
    (556,711 -> 567,398 -> 908,930)
    select sum(count) from (select count(*) as count from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and term<>'' group by lower(term) order by count(*) desc offset 10000) s;
    (2,555,158 -> 2,539,723 -> 3,697,848)
    select count(*) from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and term<>'' and array_length(regexp_split_to_array(term, ' '), 1) > 1;
    (2,812,965 -> 2,786,428 -> 4,405,141)
    select count(*) from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and term<>'';
    (3,748,417 -> 3,748,417 -> 5,552,648)
    select domain, count(*) from biosym_annotations group by domain;
    attributes | 3032462
    compounds  | 1474950
    diseases   |  829121
    mechanisms | 1444346
    --
    assignees  | 3288088
    attributes | 3021418
    compounds  | 3056458
    diseases   | 1776624
    inventors  | 3984539
    mechanisms | 3895219
    select sum(count) from (select term, count(*)  as count from biosym_annotations where term ilike '%inhibit%' group by term order by count(*) desc limit 100) s;
    (14,910 -> 15,206 -> 37,283)
    select sum(count) from (select term, count(*)  as count from biosym_annotations where term ilike '%inhibit%' group by term order by count(*) desc limit 1000) s;
    (38,315 -> 39,039 -> 76,872)
    select sum(count) from (select term, count(*)  as count from biosym_annotations where term ilike '%inhibit%' group by term order by count(*) desc offset 1000) s;
    (70,439 -> 69,715 -> 103,874)


    alter table terms ADD COLUMN id SERIAL PRIMARY KEY;
    DELETE FROM terms
    WHERE id IN
        (SELECT id
        FROM
            (SELECT id,
            ROW_NUMBER() OVER( PARTITION BY term, domain, character_offset_start, character_offset_end, publication_number
            ORDER BY id ) AS row_num
            FROM terms ) t
            WHERE t.row_num > 1 );
    """
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m scripts.patents.clean_extractions \nCleans up extracted annotations"
        )
        sys.exit()

    populate_working_biosym_annotations()
