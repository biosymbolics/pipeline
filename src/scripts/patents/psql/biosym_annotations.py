"""
To run after NER is complete
"""
import sys
import logging
from typing import Literal
from pydash import flatten

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient as DatabaseClient
from constants.core import (
    SOURCE_BIOSYM_ANNOTATIONS_TABLE as SOURCE_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE as WORKING_TABLE,
)
from utils.re import get_or_re

from .._constants import (
    COMPOUND_BASE_TERMS_GENERIC,
    MECHANISM_BASE_TERMS,
    MECHANISM_BASE_TERM_SETS,
    INTERVENTION_BASE_TERMS,
    INTERVENTION_BASE_TERM_SETS,
    INTERVENTION_BASE_PREFIXES,
)

TextField = Literal["title", "abstract"]
WordPlace = Literal["leading", "trailing", "all"]

TEXT_FIELDS: list[TextField] = ["title", "abstract"]
REMOVAL_WORDS_PRE: dict[str, WordPlace] = {
    "such": "all",
    "method": "all",
    "obtainable": "all",
    "the": "leading",
    "encoding": "trailing",
    "properties": "trailing",
    "library": "trailing",
    "more": "leading",
    "classic": "leading",
    "excellent": "all",
    "construct": "trailing",
    "particular": "leading",
    "useful": "all",
    "uses(?: thereof| of)": "all",
    "designer": "leading",
    "thereof": "all",
    "capable": "trailing",
    "specific": "leading",
    "recombinant": "leading",
    "novel": "leading",
    "human(?:ized)": "all",  # ??
    "non[ -]?toxic": "leading",
    "improved": "leading",
    "improving": "trailing",
    "new": "leading",
    "-targeted": "all",
    "long[ -]?acting": "leading",
    "potent": "trailing",
    "inventive": "leading",
    "other": "leading",
    "more": "leading",
    "of": "trailing",
    "be": "trailing",
    "use": "trailing",
    "efficacy": "trailing",
    "therapeutic procedure": "all",
    "therapeutic(?:ally)?": "leading",
    "therapeutic(?:ally)?": "trailing",
    "(?:co[ -]?)?therapy": "trailing",
    "(?:pharmaceutical |chemical )?composition": "trailing",
    "treatment method": "trailing",
    "treatment": "trailing",
    "(?:combination )?treatment": "trailing",
    "treating": "trailing",
    "component": "trailing",
    "complexe?": "trailing",
    "portion": "trailing",
    "intermediate": "trailing",
    "suitable": "all",
    "procedure": "trailing",
    "patient": "leading",
    "patient": "trailing",
    "acceptable": "all",
    "thereto": "trailing",
    "certain": "leading",
    "exemplary": "all",
    "against": "trailing",
    "usable": "trailing",
    "other": "leading",
    "suitable": "trailing",
    "pharmaceutical": "all",
    "dosage(?: form)?": "all",
    "use of": "leading",
    "certain": "leading",
    "working": "leading",
    "on": "trailing",
    "in(?: a)?": "trailing",
    "(?: ,)?and": "trailing",
    "and ": "leading",
    "the": "trailing",
    "with": "trailing",
    "a": "trailing",
    "of": "trailing",
    "for": "trailing",
    "=": "trailing",
    "unit(?:[(]s[)])?": "trailing",
    "formation": "trailing",
    "measurement": "trailing",
    "measuring": "trailing",
    "system": "trailing",
    "[.]": "trailing",
    "analysis": "trailing",
    "management": "trailing",
    "accelerated": "leading",
    "below": "trailing",
    "fixed": "leading",
    "pharmacological": "all",
    "acquisition": "trailing",
    "application": "trailing",
    "assembly": "trailing",
    "production": "trailing",
    "level": "trailing",
    "processing": "trailing",
    "lead candidate": "all",
    "control": "trailing",
    "modified": "leading",
    "variant": "trailing",
    "variety": "trailing",
    "varieties": "trailing",
    "product": "trailing",
    "family": "trailing",
    "(?:pharmaceutically|physiologically) (?:acceptable |active )?": "leading",
    "pure": "leading",
    "specific": "trailing",
    "chemically (?:modified)?": "leading",
    "based": "trailing",
    "an?": "leading",
    "active": "leading",
    "additional": "leading",
    "additive": "leading",
    "advantageous": "leading",
    "aforementioned": "leading",
    "aforesaid": "leading",
    "efficient": "leading",
    "first": "leading",
    "formula [(][ivxab]{1,3}[)]": "trailing",
    "is": "leading",
    "engineered": "leading",
    "medicament": "trailing",
    "medicinal": "leading",
    "precipitation": "trailing",
    "sufficient": "trailing",
    "due": "trailing",
    "locate": "trailing",
    "specification": "trailing",
    "detect": "trailing",
    "similar": "trailing",
    "contemplated": "leading",
    "predictable": "leading",
    "convenient": "leading",
    "dosing": "leading",
    "preferred": "leading",
    "conventional": "leading",
    "combination": "leading",
    "clinically[ -]?proven": "leading",
    "proven": "leading",
    "contemplated": "leading",
    "is indicative of": "all",
    "via": "leading",
    "effective": "all",
    "(?:high|low)[ -]?dose": "all",
    "effects of": "all",
    "soluble": "leading",
    "dosing regimen": "trailing",
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
        ]
    ]
)


DELETION_TERMS = [
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
    "detergent",
    "cellulose",
    "resin",
    "starch",
    "absorbent",
    "excipient",
    "amide",
    "amine",
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
    "detergent",
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
    "treat .*" "treating .*",
    "field of .*",
    "femur",
    "nucleotide sequence",
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
    "polynucleotide sequence",
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
        SELECT t1.publication_number AS publication_number, t2.original_term AS removal_term
        FROM {WORKING_TABLE} t1
        JOIN {WORKING_TABLE} t2
        ON t1.publication_number = t2.publication_number
        WHERE t2.original_term<>t1.original_term
        AND t1.original_term ~* CONCAT('.*', escape_regex_chars(t2.original_term), '.*')
        AND length(t1.original_term) > length(t2.original_term)
        AND array_length(regexp_split_to_array(t2.original_term, '\s+'), 1) < 3
        ORDER BY length(t2.original_term) DESC
    """

    delete_query = f"""
        DELETE FROM {WORKING_TABLE}
        WHERE ARRAY[publication_number, original_term] IN (
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
    term_sets = INTERVENTION_BASE_TERM_SETS
    prefixes = INTERVENTION_BASE_PREFIXES

    prefix_re = "|".join([p + " " for p in prefixes])

    def get_query(term_or_term_set: str | list[str], field: TextField):
        if isinstance(term_or_term_set, list):
            # term set
            re_term = "(?:" + "|".join([f"{ts}s?" for ts in term_or_term_set]) + ")"
        else:
            re_term = term_or_term_set + "s?"
        sql = f"""
            UPDATE {WORKING_TABLE} ba
            SET original_term=(substring({field}, '(?i)((?:{prefix_re})*{re_term} (?:of |for |the |that |to |comprising |(?:directed |effective |with efficacy )?against )+ (?:(?:the|a) )?.*?)(?:and|useful|for|,|$)'))
            FROM applications a
            WHERE ba.publication_number=a.publication_number
            AND original_term ~* '^(?:{prefix_re})*{re_term}$'
            AND a.{field} ~* '.*{re_term} (?:of|for|the|that|to|comprising|against|(?:directed |effective |with efficacy )?against).*'
        """
        return sql

    def get_hyphen_query(term, field: TextField):
        re_term = term + "s?"
        sql = f"""
            UPDATE {WORKING_TABLE} ba
            SET original_term=(substring(title, '(?i)([A-Za-z0-9]+-{re_term})'))
            FROM applications a
            WHERE ba.publication_number=a.publication_number
            AND original_term ~* '^{re_term}$'
            AND a.{field} ~* '.*[A-Za-z0-9]+-{re_term}.*'
        """
        return sql

    client = DatabaseClient()

    # for term in [*terms, *flatten(term_sets)]:
    #     for field in TEXT_FIELDS:
    #         try:
    #             sql = get_hyphen_query(term, field)
    #             client.execute_query(sql)
    #         except Exception as e:
    #             logging.error(e)

    # loop over term sets, in which the original_term may be in another form than the title variant
    for term_or_term_set in [*terms, *term_sets]:
        for field in TEXT_FIELDS:
            try:
                sql = get_query(term_or_term_set, field)
                client.execute_query(sql)
            except Exception as e:
                logging.error(e)


def remove_trailing_leading(removal_word_set: dict[str, WordPlace]):
    logging.info("Removing trailing/leading words")

    def get_remove_words():
        def get_sql(place):
            if place == "trailing":
                words = [
                    "[ ]" + t[0] + "s?"
                    for t in removal_word_set.items()
                    if t[1] == place
                ]
                words_re = get_or_re(words, "+")
                return f"""
                    update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '(?i){words_re}$', ''))
                    where original_term ~* '.*{words_re}$'
                """
            elif place == "leading":
                words = [
                    t[0] + "s?[ ]" for t in removal_word_set.items() if t[1] == place
                ]
                words_re = get_or_re(words, "+")
                return f"""
                    update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '(?i)^{words_re}', ''))
                    where original_term ~* '^{words_re}.*'
                """
            elif place == "all":
                words = [
                    t[0] + "s?[ ]?" for t in removal_word_set.items() if t[1] == place
                ]
                words_re = get_or_re(words, "+")
                return rf"""
                    update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '(?i)(?:^|$| ){words_re}(?:^|$| )', ' '))
                    where original_term ~* '(?:^|$| ){words_re}(?:^|$| )'
                """
            else:
                raise ValueError(f"Unknown place: {place}")

        return [get_sql(place) for place in ["all", "leading", "trailing"]]

    client = DatabaseClient()
    for sql in get_remove_words():
        client.execute_query(sql)


def remove_junk():
    """
    Remove trailing junk and silly matches
    """
    logging.info("Removing junk")

    mechanism_terms = [
        f"{t}s?"
        for t in [
            *flatten(MECHANISM_BASE_TERM_SETS),
            *MECHANISM_BASE_TERMS,
        ]
    ]
    mechanism_re = get_or_re(mechanism_terms)

    queries = [
        # unwrap
        f"update {WORKING_TABLE} "
        + r"set original_term=(REGEXP_REPLACE(original_term, '[)(]', '')) where original_term ~ '^[(][^)(]+[)]$'",
        rf"update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '^\"', '')) where original_term ~ '^\"'",
        f"update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '[)]', '')) where original_term ~ '.*[)]' and not original_term ~ '.*[(].*';",
        # leading whitespace
        rf"update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '^[ ]+', '')) where original_term ~ '^[ ]+'",
        # trailing whitespace
        rf"update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '[ ]+$', '')) where original_term ~ '[ ]+$'",
        # f"update {WORKING_TABLE} set "
        # + "original_term=regexp_extract(original_term, '(.{10,})(?:[.] [A-Z][A-Za-z0-9]{3,}).*') where original_term ~ '.{10,}[.] [A-Z][A-Za-z0-9]{3,}'",
        f"update {WORKING_TABLE} set domain='mechanisms' where domain<>'mechanisms' AND original_term ~* '.*{mechanism_re}$'",
        f"update {WORKING_TABLE} set domain='mechanisms' where domain<>'mechanisms' AND original_term in ('abrasive', 'dyeing', 'dialyzer', 'colorant', 'herbicidal', 'fungicidal', 'deodorant', 'chemotherapeutic',  'photodynamic', 'anticancer', 'anti-cancer', 'tumor infiltrating lymphocytes', 'electroporation', 'vibration', 'disinfecting', 'disinfection', 'gene editing', 'ultrafiltration', 'cytotoxic', 'amphiphilic', 'transfection', 'chemotherapy')",
        f"update {WORKING_TABLE} set domain='diseases' where original_term in ('adrenoleukodystrophy', 'stents') or original_term ~ '.* diseases?$'",
        f"update {WORKING_TABLE} set domain='compounds' where original_term in ('ethanol', 'isocyanates')",
        f"update {WORKING_TABLE} set domain='compounds' where original_term ~* '(?:^| |,)(?:molecules?|molecules? bindings?|reagents?|derivatives?|compositions?|compounds?|formulations?|stereoisomers?|analogs?|analogues?|homologues?|drugs?|regimens?|clones?|particles?|nanoparticles?|microparticles?)$' and not original_term ~* '(anti|receptor|degrade|disease|syndrome|condition)' and domain<>'compounds'",
        f"update {WORKING_TABLE} set domain='mechanisms' where original_term ~* '.*receptor$' and domain='compounds'",
        f"update {WORKING_TABLE} set domain='diseases' where original_term ~* '(?:cancer|disease|disorder|syndrome|autism|condition|psoriasis|carcinoma|obesity|hypertension|neurofibromatosis|tumor|tumour|glaucoma|arthritis|seizure|bald|leukemia|huntington|osteo|melanoma|schizophrenia)s?$' and not original_term ~* '(?:treat(?:ing|ment|s)?|alleviat|anti|inhibit|modul|target|therapy|diagnos)' and domain<>'diseases'",
        f"update {WORKING_TABLE} set domain='mechanisms' where original_term ~* '.*gene$' and domain='diseases' and not original_term ~* '(?:cancer|disease|disorder|syndrome|autism|associated|condition|psoriasis|carcinoma|obesity|hypertension|neurofibromatosis|tumor|tumour|glaucoma|retardation|arthritis|tosis|motor|seizure|bald|leukemia|huntington|osteo|atop|melanoma|schizophrenia|susceptibility|toma)'",
        f"update {WORKING_TABLE} set domain='mechanisms' where original_term ~* '.* factor$' and not original_term ~* '.*(?:risk|disease).*' and domain='diseases'",
        f"update {WORKING_TABLE} set domain='mechanisms' where original_term ~* 'receptors?$' and domain='diseases'",
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
            set original_term=substring(a.{field}, CONCAT('(?i)([^ ]*{char_set[0]}.*', escape_regex_chars(original_term), ')'))
            from applications a
            WHERE ab.publication_number=a.publication_number
            AND substring(a.{field}, CONCAT('(?i)([^ ]*{char_set[0]}.*', escape_regex_chars(original_term), ')')) is not null
            AND original_term ~* '.*{char_set[1]}.*' AND not original_term ~* '.*{char_set[0]}.*'
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
        *flatten(INTERVENTION_BASE_TERM_SETS),
        *INTERVENTION_BASE_TERMS,
    ]

    or_re = get_or_re([f"{t}s?" for t in common_terms])
    common_del_query = f"""
        delete from {WORKING_TABLE}
        where
        original_term=''
        OR original_term is null
        OR original_term ~* '^{or_re}$'
    """

    del_queries = [
        common_del_query,
        f"delete FROM {WORKING_TABLE} "
        + r"where original_term ~* '^[(][0-9a-z]{1,4}[)]?[.,]?[ ]?$'",
        f"delete FROM {WORKING_TABLE} " + r"where original_term ~ '^[0-9., ]+$'",
        f"delete FROM {WORKING_TABLE} where original_term ~* '^said .*'",
        f"delete FROM {WORKING_TABLE} where length(original_term) < 3 or original_term is null",
    ]
    for del_query in del_queries:
        DatabaseClient().execute_query(del_query)


def normalize_domains():
    """
    Normalizes domains - if the same term is used for multiple domains, pick the most common one
    """
    normalize_sql = f"""
        WITH ranked_domains AS (
            SELECT
                lower(original_term) as lot,
                domain,
                ROW_NUMBER() OVER (PARTITION BY lower(original_term) ORDER BY COUNT(*) DESC) as rank
            FROM {WORKING_TABLE}
            GROUP BY lower(original_term), domain
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
        WHERE lower(ut.original_term) = md.lot and ut.domain <> md.new_domain;
    """
    DatabaseClient().execute_query(normalize_sql)


def populate_working_biosym_annotations():
    """
    - Copies biosym annotations from source table
    - Performs various cleanups and deletions
    """
    client = DatabaseClient()
    logging.info(
        "Copying source (%s) to working (%s) table", SOURCE_TABLE, WORKING_TABLE
    )
    client.create_from_select(f"SELECT * from {SOURCE_TABLE}", WORKING_TABLE)

    # add indices after initial load
    index_base = f"index_{WORKING_TABLE}"
    client.create_indices(
        [
            {
                "table": WORKING_TABLE,
                "column": "publication_number",
            },
            {
                "table": WORKING_TABLE,
                "column": "original_term",
                "is_trgm": True,
            },
            {
                "sql": f"CREATE UNIQUE INDEX {index_base}_uniq on {WORKING_TABLE} (publication_number, original_term, domain, character_offset_start, character_offset_end)",
            },
        ]
    )

    remove_junk()
    remove_trailing_leading(
        REMOVAL_WORDS_PRE
    )  # round 1 (leaves in stuff used by for/of)
    fix_of_for_annotations()

    # round 2 (removes trailing "compound" etc)
    remove_trailing_leading(REMOVAL_WORDS_POST)
    fix_unmatched()

    remove_substrings()
    normalize_domains()
    remove_common_terms()  # final step - remove one-off generic terms


if __name__ == "__main__":
    """
    Checks:

    08/17/2023, after
    select sum(count) from (select count(*) as count from biosym_annotations where domain<>'attribute' and original_term<>'' group by lower(original_term) order by count(*) desc limit 1000) s;
    (1,038,200)
    select sum(count) from (select count(*) as count from biosym_annotations where domain<>'attribute' and original_term<>'' group by lower(original_term) order by count(*) desc offset 10000) s;
    (2,755,711)
    select count(*) from biosym_annotations where domain<>'attribute' and original_term<>'' and array_length(regexp_split_to_array(original_term, ' '), 1) > 1;
    (3,199,104)
    select count(*) from biosym_annotations where domain<>'attribute' and original_term<>'';
    (3,919,589)
    select domain, count(*) from biosym_annotations group by domain;
    attribute  | 2462930
    compounds  | 1249384
    diseases   |  840063
    mechanisms | 1830142
    select sum(count) from (select original_term, count(*)  as count from biosym_annotations where original_term ilike '%inhibit%' group by original_term order by count(*) desc limit 100) s;
    (14,419)
    select sum(count) from (select original_term, count(*)  as count from biosym_annotations where original_term ilike '%inhibit%' group by original_term order by count(*) desc limit 1000) s;
    (37,776)
    select sum(count) from (select original_term, count(*)  as count from biosym_annotations where original_term ilike '%inhibit%' group by original_term order by count(*) desc offset 1000) s;
    (71,656)
    """
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m scripts.patents.clean_extractions \nCleans up extracted annotations"
        )
        sys.exit()

    populate_working_biosym_annotations()
