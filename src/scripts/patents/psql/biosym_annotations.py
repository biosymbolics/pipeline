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
    MECHANISM_BASE_TERMS,
    MECHANISM_BASE_TERM_SETS,
    INTERVENTION_BASE_TERMS,
    INTERVENTION_BASE_TERM_SETS,
    INTERVENTION_BASE_PREFIXES,
)

TextField = Literal["title", "abstract"]
WordPlace = Literal["leading", "trailing", "all"]

TEXT_FIELDS: list[TextField] = ["title", "abstract"]
REMOVAL_WORDS: dict[str, WordPlace] = {
    "such": "all",
    "methods?": "all",
    "obtainable": "all",
    "the": "leading",
    "excellent": "all",
    "particular": "leading",
    "useful": "all",
    "uses(?: thereof| of)": "all",
    "designer": "leading",
    "thereof": "all",
    "capable": "trailing",
    "specific": "leading",
    "novel": "leading",
    "improved": "leading",
    "improving": "trailing",
    "new": "leading",
    "potent": "trailing",
    "inventive": "leading",
    "other": "leading",
    "more": "leading",
    "of": "trailing",
    "be": "trailing",
    "use": "trailing",
    "therapeutically": "trailing",
    "(?co[ -]?)?therapy": "trailing",
    "drugs?": "trailing",
    "(?:pharmaceutical |chemical )?composition": "trailing",
    "components?": "trailing",
    "complexe?s?": "trailing",
    "portions?": "trailing",
    "intermediate": "trailing",
    "suitable": "all",
    "procedure": "trailing",
    "patients?": "leading",
    "patients?": "trailing",
    "acceptable": "all",
    "thereto": "trailing",
    "certain": "leading",
    "therapeutic procedures?": "all",
    "therapeutic": "leading",
    "therapeutics?": "leading",
    "treatments?": "trailing",
    "exemplary": "all",
    "against": "trailing",
    "treatment method": "trailing",
    "(?:combination )?treatment": "trailing",
    "treating": "trailing",
    "usable": "trailing",
    "other": "leading",
    "suitable": "trailing",
    "preparations?": "trailing",
    "compositions?": "trailing",
    "combinations?": "trailing",
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
    "formations?": "trailing",
    "measurements?": "trailing",
    "measuring": "trailing",
    "systems?": "trailing",
    "[.]": "trailing",
    "analysis": "trailing",
    "methods?": "trailing",
    "management": "trailing",
    "below": "trailing",
    "fixed": "leading",
    "pharmacological": "all",
    "acquisitions?": "trailing",
    "applications?": "trailing",
    "assembly": "trailing",
    "solutions?": "trailing",
    "production": "trailing",
    "solutions?": "trailing",
    "lead candidate": "all",
    "candidate": "trailing",
    "molecules?": "trailing",
    "conjugates?": "trailing",
    "substrates?": "trailing",
    "particles?": "trailing",
    "mediums?": "trailing",
    "forms?": "trailing",
    "compounds?": "trailing",
    "control": "trailing",
    "modified": "leading",
    "variants?": "trailing",
    "variety": "trailing",
    "varieties": "trailing",
    "salts?": "trailing",
    "analogs?": "trailing",
    "analogues?": "trailing",
    "products?": "trailing",
    "family": "trailing",
    "(?:pharmaceutically|physiologically) (?:acceptable |active )?": "leading",
    "derivatives?": "trailing",
    "pure": "leading",
    "specific": "trailing",
    "chemically (?:modified)?": "leading",
    "based": "trailing",
    "an?": "leading",
    "ingredients?": "trailing",
    "active": "leading",
    "additional": "leading",
    "additives?": "leading",
    "advantageous": "leading",
    "aforementioned": "leading",
    "aforesaid": "leading",
    "candidate": "leading",
    "efficient": "leading",
    "first": "leading",
    "formula [(][ivxab]{1,3}[)]": "trailing",
    "formula": "leading",
    "formulations?": "trailing",
    "materials?": "trailing",
    "biomaterials?": "trailing",
    "is": "leading",
    "engineered": "leading",
    "medicament": "trailing",
    "medicinal": "leading",
    "variant": "leading",
    "precipitation": "trailing",
    "sufficient": "trailing",
    "due": "trailing",
    "locate": "trailing",
    "specifications?": "trailing",
    "modality": "trailing",
    "detect": "trailing",
    "similar": "trailing",
    "contemplated": "leading",
    "predictable": "leading",
    "convenient": "leading",
    "dosing": "leading",
    "dosing regimen": "trailing",
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
}

DELETION_TERMS = [
    "wherein a",
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
    "(?:.* )?apparatus(?es)?",
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
    "capsule",
    "valve",
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
    "(?.* )?tube",
    "(?.* )?lancet",
    "slider",
    "(?:.* )?tomography" "(?:.* )?instrument",
    "abutment",
    "gasket",
    "(?:.* )?wave",
    "(?:.* )?pump",
    "(?:.* )?article",
    "(?:.* )?screw)",
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
    "specific antibodies",
    "catalytic",
    "gene delivery",
    "said protein",
    "fibrous",
    "in vitro",
    "polypeptides present",
    "volatile",
    "amino acid",
    "human(?:ized)? antibody",
    "human(?:ized)? antibodies",
    "silica",
    # procedure
    "(?.* )?ablation",
    "(?:.* )?surger(?:y|ies)",
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
    "compound[(]s[])]",
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
    "pharmacologically active agents",
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
    ".*patients",
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
    "keratin fib(?:re|er)s?)",
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
    """
    query = f"""
        CREATE OR REPLACE TABLE names_to_remove AS
            SELECT t1.publication_number AS publication_number, t2.original_term AS removal_term
            FROM {WORKING_TABLE} t1
            JOIN {WORKING_TABLE} t2
            ON t1.publication_number = t2.publication_number
            WHERE t2.original_term<>t1.original_term
            AND lower(t1.original_term) like CONCAT('%', lower(t2.original_term), '%')
            AND length(t1.original_term) > length(t2.original_term)
            AND array_length(SPLIT(t2.original_term, ' ')) < 3
            ORDER BY length(t2.original_term) DESC
    """

    delete_query = f"""
        DELETE FROM {WORKING_TABLE}
        WHERE (publication_number, original_term) IN (
            SELECT (publication_number, removal_term)
            FROM names_to_remove
        )
    """

    client = DatabaseClient()

    client.execute_query(query)
    client.execute_query(delete_query)
    client.delete_table("names_to_remove")


def fix_of_for_annotations():
    """
    Handles "inhibitors of XYZ" and the like, which neither GPT or SpaCyNER were good at finding
    (but high hopes for binder)
    """
    # Define terms

    terms = INTERVENTION_BASE_TERMS
    term_sets = INTERVENTION_BASE_TERM_SETS
    prefixes = INTERVENTION_BASE_PREFIXES

    prefix_re = "|".join([p + " " for p in prefixes])

    def get_query(term_or_term_set: str | list[str], field: TextField):
        if isinstance(term_or_term_set, list):
            # term set
            re_term = "(?:" + "|".join([f"{ts}s?" for ts in term_or_term_set]) + ")"
        else:
            re_term = term + "s?"
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
    for term in terms:
        for field in TEXT_FIELDS:
            sql = get_query(term, field)
            client.execute_query(sql)

    for term in [*terms, *[t for term_set in term_sets for t in term_set]]:
        for field in TEXT_FIELDS:
            sql = get_hyphen_query(term, field)
            client.execute_query(sql)

    # loop over term sets, in which the original_term may be in another form than the title variant
    for term_set in term_sets:
        for field in TEXT_FIELDS:
            sql = get_query(term_set, field)
            client.execute_query(sql)


def remove_junk():
    """
    Remove trailing junk and silly matches
    """

    def get_remove_words():
        def get_sql(place):
            if place == "trailing":
                words = [
                    "[ ]" + t[0] + "s?" for t in REMOVAL_WORDS.items() if t[1] == place
                ]
                words_re = get_or_re(words, "+")
                return f"""
                    update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '(?i){words_re}$', ''))
                    where original_term ~* '.*{words_re}$'
                """
            elif place == "leading":
                words = [t[0] + "s?[ ]" for t in REMOVAL_WORDS.items() if t[1] == place]
                words_re = get_or_re(words, "+")
                return f"""
                    update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '(?i)^{words_re}', ''))
                    where original_term ~* '^{words_re}.*'
                """
            elif place == "all":
                words = [
                    t[0] + "s?[ ]?" for t in REMOVAL_WORDS.items() if t[1] == place
                ]
                words_re = get_or_re(words, "+")
                return rf"""
                    update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '(?i)(?:^|$| ){words_re}(?:^|$| )', ' '))
                    where original_term ~* '(?:^|$| ){words_re}(?:^|$| )'
                """
            else:
                raise ValueError(f"Unknown place: {place}")

        return [get_sql(place) for place in ["leading", "trailing", "all"]]

    delete_term_re = "^" + get_or_re([f"{dt}s?" for dt in DELETION_TERMS]) + "$"
    mechanism_terms = [
        f"{t}s?"
        for t in [
            "anaesthetic",
            *flatten(MECHANISM_BASE_TERM_SETS),
            *MECHANISM_BASE_TERMS,
        ]
    ]
    mechanism_re = get_or_re(mechanism_terms)

    queries = [
        f"update `{WORKING_TABLE}` "
        + r"set original_term=(REGEXP_REPLACE(original_term, '[)(]', '')) where original_term ~ '^[(][^)(]+[)]$'",
        *get_remove_words(),
        f"update `{WORKING_TABLE}` "
        + "set original_term=(REGEXP_REPLACE(original_term, '[ ]{2,}', ' ')) where original_term ~ '[ ]{2,}'",
        f"update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '^[ ]+', '')) where original_term ~ '^[ ]+'",
        f"update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '[ ]$', '')) where original_term ~ '[ ]$'",
        f"update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, r'^\"', '')) where original_term ~ r'^\"'",
        f"update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '[)]', '')) where original_term like '%)' and original_term not like '%(%';",
        f"update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, 'disease factor', 'disease')) where original_term like '% disease factor';",
        f"update `{WORKING_TABLE}` set "
        + "original_term=regexp_extract(original_term, '(.{10,})(?:[.] [A-Z][A-Za-z0-9]{3,}).*') where original_term ~ '.{10,}[.] [A-Z][A-Za-z0-9]{3,}'",
        f"delete FROM `{WORKING_TABLE}` "
        + r"where original_term ~ '^[(][0-9a-z]{1,4}[)]?[.,]?[ ]?$'",
        f"delete FROM `{WORKING_TABLE}` " + r"where roriginal_term ~ '^[0-9., ]+$'",
        f"delete FROM `{WORKING_TABLE}` where original_term like 'said %'",
        f"delete from `{WORKING_TABLE}` where domain='compounds' AND (original_term ~* '.*(?:.*tor$)') and not original_term ~* '(?:vector|factor|receptor|initiator|inhibitor|activator|ivacaftor|oxygenator|regulator)')",
        f"delete FROM `{WORKING_TABLE}` where length(original_term) < 3 or original_term is null",
        f"delete from `{WORKING_TABLE}` where original_term ~* {delete_term_re}",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where domain<>'mechanisms' AND original_term ~* '.*{mechanism_re}$'",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where domain<>'mechanisms' AND original_term in ('abrasive', 'dyeing', 'dialyzer', 'colorant', 'herbicidal', 'fungicidal', 'deodorant', 'chemotherapeutic',  'photodynamic', 'anticancer', 'anti-cancer', 'tumor infiltrating lymphocytes', 'electroporation', 'vibration', 'disinfecting', 'disinfection', 'gene editing', 'ultrafiltration', 'cytotoxic', 'amphiphilic', 'transfection', 'chemotherapy')",
        f"update `{WORKING_TABLE}` set domain='diseases' where original_term in ('adrenoleukodystrophy', 'stents') or original_term ~ '.* diseases?$'",
        f"update `{WORKING_TABLE}` set domain='compounds' where original_term in ('ethanol', 'isocyanates')",
        f"update `{WORKING_TABLE}` set domain='compounds' where original_term ~* '(?:^| |,)(?:molecules?|molecules? bindings?|reagents?|derivatives?|compositions?|compounds?|formulations?|stereoisomers?|analogs?|analogues?|homologues?|drugs?|regimens?|clones?|particles?|nanoparticles?|microparticles?)$' and not original_term ~* '(anti|receptor|degrade|disease|syndrome|condition)' and domain<>'compounds'",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where original_term like '% receptor' and domain='compounds'",
        f"update `{WORKING_TABLE}` set domain='compounds' where original_term like '% acid' and domain='mechanism'",
        f"update `{WORKING_TABLE}` set domain='compounds' where original_term like '%quinolones' and domain='mechanism'",
        f"update `{WORKING_TABLE}` set domain='compounds' where original_term='manganese' and domain<>'compounds'",
        f"update `{WORKING_TABLE}` set domain='diseases' where original_term ~* '(?:cancer|disease|disorder|syndrome|autism|condition|psoriasis|carcinoma|obesity|hypertension|neurofibromatosis|tumor|tumour|glaucoma|arthritis|seizure|bald|leukemia|huntington|osteo|melanoma|schizophrenia)s?$' and not original_term ~* '(?:treat(?:ing|ment|s)?|alleviat|anti|inhibit|modul|target|therapy|diagnos)' and domain<>'diseases'",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where original_term like '% gene' and domain='diseases' and not original_term ~* '(?:cancer|disease|disorder|syndrome|autism|associated|condition|psoriasis|carcinoma|obesity|hypertension|neurofibromatosis|tumor|tumour|glaucoma|retardation|arthritis|tosis|motor|seizure|bald|leukemia|huntington|osteo|atop|melanoma|schizophrenia|susceptibility|toma)'",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where original_term like '% factor' and original_term not like '%risk%' and original_term not like '%disease%' and domain='diseases'",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where original_term ~ 'receptors?$' and domain='diseases'",
    ]
    client = DatabaseClient()
    for sql in queries:
        client.execute_query(sql)


def fix_unmatched():
    """
    Example: 3 -d]pyrimidine derivatives -> Pyrrolo [2, 3 -d]pyrimidine derivatives
    """

    def get_query(field, char_set):
        sql = rf"""
            UPDATE {WORKING_TABLE} ab
            set original_term=substring(a.{field}, CONCAT(r'(?i)([^ ]*\{char_set[0]}.*', escape_regex_chars(original_term), ')'))
            from applications a
            WHERE ab.publication_number=a.publication_number
            AND substring(a.{field}, CONCAT(r'(?i)([^ ]*\{char_set[0]}.*', escape_regex_chars(original_term), ')')) is not null
            AND original_term like '%{char_set[1]}%' AND original_term not like '%{char_set[0]}%'
            AND {field} like '%{char_set[0]}%{char_set[1]}%'
        """
        return sql

    client = DatabaseClient()
    for field in TEXT_FIELDS:
        for char_set in [("{", "}"), ("[", "]"), ("(", ")")]:
            sql = get_query(field, char_set)
            client.execute_query(sql)


def remove_common_terms():
    """
    Remove common original terms
    """
    # regex in here, effectively ignored
    common_terms = [
        *flatten(INTERVENTION_BASE_TERM_SETS),
        *INTERVENTION_BASE_TERMS,
    ]
    with_plurals = [
        *common_terms,
        *[f"{term}s" for term in common_terms],
    ]

    # hack regex check
    str_match = ", ".join(
        [f"'{term.lower()}'" for term in with_plurals if "?" not in term]
    )
    re_match = " OR ".join(
        [
            f"original_term ~ '^{term.lower()}s?$'"
            for term in common_terms
            if "?" in term
        ]
    )
    query = f"delete from {WORKING_TABLE} where lower(original_term) in ({str_match}) OR {re_match}"
    DatabaseClient().execute_query(query)


def create_working_biosym_annotations():
    """
    - Copies biosym annotations from source table
    - Performs various cleanups and deletions
    """
    client = DatabaseClient()
    logging.info(
        "Copying source (%s) to working (%s) table", SOURCE_TABLE, WORKING_TABLE
    )
    client.select_to_table(f"SELECT * from {SOURCE_TABLE}", WORKING_TABLE)

    # add indices after initial load
    client.add_indices(
        [
            f"CREATE INDEX index_publication_number on {WORKING_TABLE} (publication_number)",
            f"CREATE INDEX trgm_index_original_term on {WORKING_TABLE} USING gin (lower(original_term) gin_trgm_ops)",
        ]
    )

    fix_of_for_annotations()
    fix_unmatched()

    remove_junk()
    remove_substrings()
    remove_common_terms()  # final step - remove one-off generic terms


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m scripts.patents.clean_extractions \nCleans up extracted annotations"
        )
        sys.exit()

    create_working_biosym_annotations()