"""
To run after NER is complete
"""
import sys
import logging
from typing import Literal
from pydash import compact, flatten
import polars as pl

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
from core.ner.cleaning import EntityCleaner
from utils.re import expand_res, get_or_re


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
    "good": "all",
    "target": "leading",
    "properties": "trailing",
    "derived": "all",
    "library": "all",
    "more": "leading",
    "technique": "trailing",
    "classic": "all",
    "present": "leading",
    "invention": "all",
    "excellent": "all",
    "super": "all",  # ??
    "optimal": "all",
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
    "pharmaceutical compositions?(?: comprising)?": "all",
    "therapeutic procedure": "all",
    "therapeautic?": "all",
    "therapeutic(?:ally)?": "all",
    "therefor": "all",
    "prophylactic": "all",
    "(?:co[ -]?)?therapy": "trailing",
    "(?:pharmaceutical |chemical )?composition": "trailing",
    "(?:pre[ -]?)?treatment (?:method|with|of)": "all",
    "treating": "all",
    "contact": "trailing",
    "portion": "trailing",
    "intermediate": "all",
    "suitable": "all",
    "and uses thereof": "all",
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
    "diagnosis": "all",
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
    "engineered": "leading",
    "engineered": "trailing",
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
    "effect": "trailing",
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
    "[(][0-9a-z]{1,4}[)]?[.,]?[ ]?",
    "[0-9., ]+",  # del if only numbers . and ,
    # OR length(original_term) > 150 and original_term ~* '\y(?:and|or)\y' -- del if sentence
    # OR length(original_term) > 150 and original_term ~* '.*[.;] .*' -- del if sentence
    # mangled
    "optionally other modification",
    "co[- ]?operate",
    "light[ -]?receiving",
    "structure directing",
    "resultant",
    "optionally other",
    "above[ -]?mentioned",
    ".* of",
    ".* comprising",
    ".* tubular",
    "composition (?:contain|compris)",
    "by-",
    "cur",
    "co-",
    "therefrom",
    "said .*",
    "quantify",
    "citrat",
    "functional",
    "cross[ -]?linking",
    "biocompatibility",
    "particle diamet",
    "embodiment",
    "quantificat",
    "(?:.* )?rate",
    "correspond",
    "water[- ]?soluble",
    "illustrat",
    "cross[- ]?link",
    "antigen-binding",
    "sterilization",
    "regeneration",
    "particulat",
    "embodiment",
    "least partial",
    "reform",
    "conformat",
    # thing (unwanted because generic)
    "carbon dioxide",  # generic
    "product .*",
    "pathogen(?:ic)?",
    "regenerative medicine",
    "basic amino acid",
    "(?:.* )?characteristic",
    "response",
    "single piece",
    "product gas",
    r"agent\(s\)",
    "byproduct",
    "medical purpose",
    "cell membrane",
    "(?:.* )?part",
    "product mixture",  # single dose
    "(?:.* )?activit",
    "(?:.* )?member",
    "module",
    "group",
    "product stream",
    "operator",
    "field of .*",
    # thing (unwanted because wrong type of thing)
    "aromatic ring",  # industrial
    "thermoplastic polymer",  # industrial
    "polymer matrix",  # industrial
    "polyolefin",  # industrial
    "polyisocyanate",  # industrial
    "alkaline",  # industrial
    "trifluoromethyl",  # industrial
    "(?:.* )?arrangement",
    "(?:.* )?position",
    "(?:.* )?frame",
    "(?:.* )?feature",
    "heat generat",
    "(?:.* )?industry",
    "impact modifier",
    "scratch resistance",
    "gene transfer",
    "mother milk",
    "drug development",
    ".*administration",
    ".*patient",
    "(?:.* )?pest",
    "(?:.* )?examinee",
    "(?:.* )?staff",
    "(?:.* )?trial",
    "(?:.* )?infant",
    "(?:.* )?prospect",
    "(?:.* )?room",
    "(?:.* )?personnel",
    "body part",
    "(?:.* )?patent",
    "(?:.* )?pathway",
    "pest control",
    "(?:.* )?animal",
    "(?:.* )?retardant",  # e.g. flame retardant
    "aroma",
    "(?:.* )?reaction",
    "(?:.* )?cosmetic",
    "(?:.* )?fragrance",
    "ethylene",
    "alkylat",
    "carbonyl",
    "aldehyde",
    "silica",
    "keratin",
    "perfum",
    "propane",
    "bacteri(?:um|a)",
    "(?:.*)?station",
    "(?:.* )?location",
    "(?:.* )?mode",
    "(?:.* )?region",
    "(?:tumou?r|eukaryotic|normal|animal|bacterial|single|skin|cancerous|insect) cell",
    "virus",  # generic
    "titanium dioxide",
    "microorganism",
    "(?:.* )?metal",  # temporary?
    "(?:.* )?area",
    "(?:.* )?power",
    "(?:.* )?site",
    "(?:.* )?signal",
    "(?:.* )?layer",
    "(?:.* )?surface",  # device?
    "thermoplastic",
    # effect
    "detrimental",
    "(?:.* )?absorb",
    "(?:.* )?disengage",
    "adjustment",
    "lubricat",
    "chain transfer",
    "(?:.* )?abrasive",
    "(?:.* )?retardancy",  # e.g. flame retardant
    "film[ -]?form",
    "heat transfer",
    "nucleating",
    "cell death",
    "deformable",
    "(?:.* )?growth",  # TODO: disease?
    "(?:.* )?resistance",
    "(?:.* )?addition",
    "ameliorat",
    "transformant",
    "deformation",
    "reduction",
    "fixation",
    "stiffen",
    "suction",
    # disease
    "disease state",
    "dysfunction",
    "related (?:disorder|condition|disease)",
    "(?:disorder|condition|disease)s related",
    "disease",
    "disorder",
    "dysfunction",
    # part or characteristic of body, or fluid
    "(?:.* )?urine",
    "(?:.* )?appendage",
    "(?:.* )?ventricle",
    "(?:.* )?aorta",
    "(?:.* )?artery",
    "(?:.* )?vein",
    "(?:.* )?tissue",
    "(?:.* )?septum",
    "(?:.* )?vasculature",
    "(?:.* )?tendon",
    "(?:.* )?ligament",
    "bone fragment",
    "vertebral",
    "(?:.* )?obturator",
    "(?:.* )?atrium",
    "tibial?",
    "transdermal",
    "(?:.* )?femur",
    "(?:.* )?core",  # device
    "(?:.* )?vesicle",
    "(?:.* )?valve",  # device
    "(?:.* )?atrium",
    "(?:.* )?nail",  # device
    "(?:.* )?joint",  # treatment of diseases.
    "urea",
    "gastrointestinal(?: tract)?",
    "cartilage",
    "jaw",
    "liver",
    "heart",
    "intraocular len",
    "femoral",
    "respiratory(?: tract)?",
    "pulmonary",
    "uterus",
    "lung",
    "spinal column",
    "muscle",
    "kidney",
    "prostate",
    "pancreas",
    "ocular",
    "spleen",
    "(?:.* )?cavity",
    "gallbladder",
    "bladder",
    "ureter",
    "urethra",
    "esophagus",
    "stomach",
    "intestin(?:e|al)",
    "colon",
    "rectum",
    "trachea",
    "bronch(?:ial|us)",
    "larynx",
    "pharynx",
    "nasal",
    "sinus",
    "arterial",
    "venous",
    "lymph(?:atic)?(?: node| vessel)?",
    # category errors
    "engaged position",
    "cell[- ]?free",
    "contribut",
    "advantage",
    "cell cycle",
    "cell wall",
    "(?:.* )?side",  # base end side
    "accept",
    "solid state",
    "susceptible",
    "(?:.* )?typ",  # type
    "processability",
    "auxiliary",
    "present tricyclic",
    "wherein(?: a)?",
    "particle size",
    "fragment of",
    "concentric(?:al)?",
    "(?:.* )?coupled",
    "(?:.* )?axis",
    "(?:.* )?path",
    "(?:.* )?pattern",
    "(?:.* )?plan",
    "(?:.* )?stage",  # ??
    "(?:.* )?quantit",
    "functional group",
    "drug discovery",
    "combinatorial",
    "compound having",
    "non-",
    # generic
    "design",
    "general structural",
    "piece",
    "attribute",
    "preform",
    "DNA sequence",
    "(?:medical|treatment) (?:fluid|zone|container|section|technology)",
    "object",
    "(?:drug )?target",
    "biologically active",
    "(?:.* )?sample",
    "therapeutically",
    "pharmaceutically-",
    "physiological",
    "pharmaceutically[- ]accept",
    "therapy",
    "(?:general |chemical )?formula(?: i{1,3}){0,3}(?: [0-9a-z]){0,3}",
    "components?(?: i{1,3})?(?: [0-9a-z])?",
    "compounds?(?: i{1,3})?(?: [0-9a-z])?",
    "(?:medicinal|bioactive|biological)(?: activity)?",
    "(?:pharmaceutical|pharmacological|medicinal) composition(?: containing| comprising)?",
    "medicine compound",
    "wherein said compound",
    "pharmaceutical",
    "pharmacological",
    "(?:.* )?compounds? of(?: formula)?",
    "compound of",
    "(?:pharmacologically|pharmaceutically|biologically) active agent",
    "(?:bioactive |medicinal )?agent",
    "chemical entit",
    # general characteristic
    "bioavailability",
    "usefulness",
    "(?:.* )?conductivit",
    "multi[ -]?function",
    "symmetric(?:al)?",
    "biocompatible",
    "bioactivit",
    "medicinal",
    "cellular",
    "natural",
    "substantially free",
    "therapeutically active",
    # characteristics / descriptors
    "(?:non[ -]?)?conductive",
    "granular",
    "luminescent",
    "(?:.* )?bound",
    "amorphous",
    "purif",  # purified
    "hardness",
    "deactivate",  # ??
    "(?:.* )?stability",
    "planar",
    "fatty",
    "compressible",
    "(?:.* )?bond",
    "(?:.* )?perpendicular",
    "(?:.* )?specificity",
    "(?:.* )?domain",
    "storage stability",
    "conductive",
    "flexible",
    "dermatological",
    "bifunctional",
    "in vitro",
    "volatile",
    "fibrous",
    "biodegrad",
    "resilient",
    "fluorescent",
    "superabsorbent",
    "nonwoven",
    "crystalline",
    "edibl",
    "non[ -]?therapeutic",
    "water[ -]?insoluble",
    "unsaturat",
    "adhes",
    "porous",
    "dispens",
    "impedanc",
    "radioact",
    "cleansing",
    "optionally substitut",
    "non[ -]?invasive",
    "reinforc",
    "single chain",
    "aqueous",
    "single bond",
    "concentration",
    "(?:.* )?conductive",
    "recombinant",
    "genetic",
    "acidic",
    "unsubstituted",
    "gaseous",
    "phenolic",
    "aromatic",
    "conjugated",
    "polymeric",
    "inorganic",
    "heterocyclic",
    "oligomeric",
    "synergistic",
    "immunogenic",
    "macrocyclic",
    "elastic",
    "catalytic",
    "hydrophilic",
    "ophthalmic",
    "heterocyclic",
    "hydrophobic",
    "enzymatic",
    "lipophilic",
    "(?:.* )?biodegradabilit",
    "(?:.* )?affinity",
    "residual",
    "rigid",
    "cyclic",
    "adverse",
    # physical process
    "drug release",
    "sustained[ -]?release",
    "disintegrat",
    "evaporat",
    "agglomerat",
    # measurable thing
    "(?:.* )?torque",
    "(?:.* )?weight",
    "(?:.* )?wavelength",
    "(?:.* )?charge",
    "(?:.* )?band",
    "(?:.* )?viscosity",
    "(?:.* )?volume",
    "(?:.* )?distribution",
    "(?:.* )?amount",
    "refractive index",
    "phase change",
    "(?:.* )?pressure",
    "(?:.* )?strength",
    "(?:.* )?population",
    "(?:.* )?end",
    "(?:.* )?temperature",
    "(?:.* )?ratio",
    "(?:.* )?deliver",
    "(?:.* )?step",
    "(?:.* )?value",
    "(?:.* )?time",
    # body stuff
    "plasma",
    # roa
    "aerosol(?:[- ]?forming)?",
    "parenteral",
    "inhalation",
    "insufflation",
    "oral",
    "intranasal",
    "intramuscular",
    "intravenous",
    "subcutaneous",
    "topical",
    # intangible thing
    "(?:.* )?data",
    "(?:.* )?integer",
    "(?:.* )?language",
    "(?:.* )?information",
    "(?:.* )?report",
    "(?:.* )?parameter",
    "(?:.* )?constant",
    "(?:.* )?longitudinal",
    "(?:.* )?equivalent",
    "(?:.* )?subset",
    "time period",
    "(?:.* )?memory",
    # material
    "(?:.* )?metal",
    # device
    "(?:.* )?pacemaker",
    "(?:.* )?glass",
    "(?:.* )?linkage",  # ??
    "(?:.* )?rubber",  # ??
    "(?:.* )?tank",
    "(?:.* )?computer.*",
    "(?:.* )?latch",
    "(?:.* )?manifold",
    "(?:.* )?clip",
    "(?:.* )?belt",
    "(?:.* )?pivot",
    "(?:.* )?mask",
    "(?:.* )?board",
    "(?:.* )?bridge",
    "(?:.* )?cuff",
    "(?:.* )?pouch",
    "(?:.* )?container",
    "mobile",
    "(?:.* )?fiber",  # TODO could be bio
    "(?:.* )?conductor",
    "(?:.* )?connector",
    "(?:.* )?effector",
    "(?:.* )?head",
    "(?:.* )?tape",
    "(?:.* )?inlet",
    "(?:.* )?outlet",
    "(?:.* )?source",  # TODO
    "(?:.* )?strip",  # e.g. test strip
    "core[ -]?shell",  # what is this?
    "stop(?:per)?",
    "(?:.* )?window",
    "(?:.* )?solid state",
    "(?:.*)?wire",  # e.g. guidewire
    "(?:.* )?bed",
    "(?:.* )?prosthetic",
    "(?:.* )?equipment",
    "(?:.* )?generator",
    "(?:.* )?(?:micro)?channel",
    "(?:.* )?light[ -]?emitt(?:er|ing)s?.*",
    "(?:.* )?cathode",
    "(?:.* )?dielectric",
    "(?:.* )?mandrel",
    "(?:.* )?stylet",
    "(?:.* )?coupling",
    "(?:.* )?attachment",
    "(?:.* )?shaft",
    "(?:.* )?body",
    "(?:.* )?aperture",
    "(?:.* )?biosensor",
    "(?:.* )?conduit",
    "(?:.* )?sheath",
    "(?:.* )?compartment",
    "(?:.* )?receptacle",
    "(?:.* )?endoscope",
    "(?:.* )?article",
    "(?:.* )?nozzle",
    "(?:.* )?plastic",
    "(?:.* )?table",
    ".*mechanical.*",
    "(?:.* )?holder",
    "(?:.* )?circuit",
    "(?:.* )?liner",
    "(?:.* )?paper",
    "(?:.* )?light",
    "(?:.* )?solar cell.*",
    "(?:.* )?ground",
    "(?:.* )?waveform",
    "(?:.* )?tool",
    "(?:.* )?centrifug",
    "(?:.* )?centrifugat",
    "(?:.* )?current",
    "(?:.* )?surfac",
    "(?:.* )?field",
    "(?:.* )?mou?ld",  # molding, moulded
    "(?:.* )?napkin",
    "(?:.* )?display",
    "(?:.*[ -])?scale",
    "(?:.* )?imag",  # imaging
    "(?:.* )?port",
    "(?:.* )?seperat",
    "(?:.* )?(?:bio)?reactor",
    "(?:.* )?program",
    "(?:.* )?plat",  # plate
    "(?:.* )?vessel",
    "(?:.* )?device",
    "(?:.* )?motor",
    "(?:.* )?(?:bio[ -]?)?film",
    "(?:.* )?instrument",
    "(?:.* )?hinge",
    "(?:.* )?dispens",
    "(?:.* )?tip",
    "(?:.* )?prob",  # probe
    "(?:.* )?rod",
    "(?:.* )?prosthesis",
    "(?:.* )?catheter.*",
    "(?:.* )?trocar",
    "(?:.* )?electrode",
    "(?:.* )?fasten",
    "(?:.* )?waveguide",
    "(?:.* )?spacer",
    "(?:.* )?radiat",
    "(?:.* )?implant",
    "(?:.* )?actuat",
    "(?:.* )?clamp",
    "(?:.* )?spectromet",
    "(?:.* )?fabric",
    "(?:.* )?diaper",
    "(?:.* )?coil",
    "(?:.* )?apparatus(?:es)?",
    "(?:.* )?sens",  # sensor
    "(?:.* )?waf",  # wafer
    "(?:.* )?tampon",
    "(?:.* )?(?:top)?sheet",
    "(?:.* )?pad",
    "(?:.* )?syringe(?: .*)?",
    "(?:.* )?canist",
    "(?:.* )?tether",
    "(?:.* )?camera" "(?:.* )?mouthpiec",
    "(?:.* )?transduc",
    "electrical stimulat",
    "(?:.* )?toothbrush",
    "(?:.* )?strut",
    "(?:.* )?sutur",
    "(?:.* )?cannula",
    "(?:.* )?stent",
    "(?:.* )?capacit",
    "acceleromet",
    "contain",
    "(?:.* )?reservoir",
    "(?:.* )?housing",
    "(?:.* )?inject",
    "(?:.* )?diaphragm",
    "(?:.* )?cartridge",
    "(?:.* )?plunger",
    "(?:.* )?ultrasound(?: .*)?",
    "(?:.* )?piston",
    "(?:.* )?balloon",
    "(?:.* )?stapl",
    "(?:.* )?engine",
    "(?:.* )?gasket",
    "(?:.* )?wave",
    "(?:.* )?pump",
    "(?:.* )?article",
    "(?:.* )?screw",
    "(?:.* )?cytomet",
    "(?:.* )?appliance",
    "(?:.* )?blow",
    "(?:.* )?tube",
    "(?:.* )?lancet",
    "(?:.* )?capsul",
    "(?:.* )?valv",
    "(?:.* )?bladder",
    "solub",
    "(?:.* )?compressor",
    "(?:.* )?forcep",
    "(?:.* )?splitt",  # splitter
    "(?:.* )?battery",
    "(?:.* )?blad",  # blade
    "(?:.* )?needl",
    "(?:.* )?wheelchair",
    "(?:.* )?machine",
    "(?:.* )?applicat",
    "(?:.* )?monit",
    "(?:.* )?irrigat",
    "(?:.* )?accelerat",
    "(?:.* )?indicat",
    "(?:.* )?pump",
    "(?:.* )?chamber",
    "(?:.* )?sponge",
    "(?:.* )?textile",
    "(?:.* )?lead",
    "(?:.* )?block",  # TODO: procedure?
    ".*graphy",
    ".*transceiver",
    ".*piezoelectric.*",
    ".*ultrasonic.*",
    "impeller",
    "transmit",
    "slider",
    "abutment",
    "interferometer",
    "fastening mean",
    "piezoelectric.*",
    "handpiece",
    "reagent kit",
    "(?:.* )?diode",
    "anvil",
    "centrifugal force",
    "implant .*",
    "robot.*",
    "(?:fuel|electrochemical) cell",
    # "(?:.* )?generator",??
    # end device
    # non-medical procedure or process
    "pattern form",  # pattern forming
    "crystallizat",
    "quantitat",
    "(?:administrative )?procedure",
    "support (?:.*)",  # ??
    "(?:.* )?communicat",
    "(?:.* )?sequenc",
    "(?:.* )?(?:micro)?process",
    # procedure
    "(?:.* )?electrolysis",
    "(?:.* )?incision",
    "(?:.* )?graft",
    "prognosticat",
    "(?:.* )?ablation",
    "(?:.* )?technique",
    "(?:.* )?retract",
    "(?:.* )?care",
    "(?:.* )?amplification",
    "(?:.* )?ablation",
    "(?:.* )?surger(?:y|ie)",
    "brachytherapy",
    "radiotherapy",
    "sealant",
    "microelectronic",
    # agtech
    "(?:.* )?feedstock",
    "(?:.* )?biomass",
    "(?:.* )?agrochemical",
    "(?:.* )?herbicid(?:e|al)(?: activity)?",
    "(?:.* )?insecticid(?:e|al)(?: activity)?",
    "(?:.* )?biocid(?:e|al)(?: activity)?",
    "(?:.* )?fungicid(?:e|al)(?: activity)?",
    "(?:.* )?pesticid(?:e|al)(?: activity)?",
    "(?:.* )?plant",
    "drought tolerance",
    "biofuel",
    "biodiesel",
    "plant growth regulator",
    # end agtech
    "explosive",
    # diagnostic
    "(?:.* )?polymerase chain reaction",
    "(?:.* )?testing",
    "(?:.* )?detect",
    "(?:.* )?diagnostic",
    "(?:.* )?diagnosis",
    "analyt",
    "(?:.* )?scopy" "(?:.* )?reagent",
    # end diagnostic
]


def remove_substrings():
    """
    Removes substrings from annotations
    (annotations that are substrings of other annotations for that publication_number)

    Note: good to run pre-training
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
    prefixes = INTERVENTION_PREFIXES

    prefix_re = "|".join([p + " " for p in prefixes])

    # e.g. inhibition of apoptosis signal-regulating kinase 1 (ASK1)
    def get_query(re_term: str, field: TextField):
        sql = f"""
            UPDATE {WORKING_TABLE} ba
            SET original_term=(substring({field}, '(?i)((?:{prefix_re})*{re_term} (?:of |for |the |that |to |comprising |(?:directed |effective |with efficacy )?against )+(?:(?:the|a) )?.*?)(?:and|useful|for|,|.|$)'))
            FROM applications a
            WHERE ba.publication_number=a.publication_number
            AND original_term ~* '^(?:{prefix_re})*{re_term}$'
            AND a.{field} ~* '.*{re_term} (?:of|for|the|that|to|comprising|(?:directed |effective |with efficacy )?against).*'
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
                    update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '\y{words_re}$', '', 'gi'))
                    where original_term ~* '.*\y{words_re}$'
                """
            elif place == "leading":
                words = [
                    t[0] + r"s?[ ]*" for t in removal_word_set.items() if t[1] == place
                ]
                if len(words) == 0:
                    return None
                words_re = get_or_re(words, "+")
                return rf"""
                    update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '^{words_re}\y', '', 'gi'))
                    where original_term ~* '^{words_re}\y.*'
                """
            elif place == "all":
                words = [
                    t[0] + "s?[ ]*" for t in removal_word_set.items() if t[1] == place
                ]
                if len(words) == 0:
                    return None
                words_re = get_or_re(words, "+")
                return rf"""
                    update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '\y{words_re}\y', ' ', 'gi'))
                    where original_term ~* '.*\y{words_re}\y.*'
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
        # removes evertyhing after a newline
        rf"update {WORKING_TABLE} set original_term= regexp_replace(original_term, '\.?\s*\n.*', '') where  original_term ~ '.*\n.*'",
        # unwrap
        f"update {WORKING_TABLE} "
        + r"set original_term=(REGEXP_REPLACE(original_term, '[)(]', '', 'g')) where original_term ~ '^[(][^)(]+[)]$'",
        rf"update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '^\"', '')) where original_term ~ '^\"'",
        # orphaned closing parens
        f"update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '[)]', '')) "
        + "where original_term ~ '.*[)]' and not original_term ~ '.*[(].*';",
        # leading/trailing whitespace
        rf"update {WORKING_TABLE} set original_term=trim(original_term) where trim(original_term) <> original_term",
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
    client = DatabaseClient()
    common_terms = [
        *DELETION_TERMS,
        *INTERVENTION_BASE_TERMS,
        *EntityCleaner().common_words,
    ]

    common_terms_re = get_or_re([f"{t}?" for t in common_terms])
    del_term_res = [
        # .? - to capture things like "gripping" from "grip"
        f"^{common_terms_re}.?(?:ing|e|ied|ed|er|or|en|ion|ist|ly|able|ive|al|ic|ous|y|ate|at|ry|y|ie)*s?$",
    ]
    del_term_re = "(?i)" + get_or_re(del_term_res)
    print(del_term_re)
    result = client.select(f"select distinct original_term from {WORKING_TABLE}")
    terms = pl.Series([r.get("original_term") for r in result])

    delete_terms = terms.filter(terms.str.contains(del_term_re)).to_list()

    del_query = f"""
        delete from {WORKING_TABLE}
        where original_term=ANY(%s)
        or original_term is null
        or original_term = ''
        or length(trim(original_term)) < 3
    """
    DatabaseClient().execute_query(del_query, (delete_terms,))


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
    for sql in queries:
        client.execute_query(sql)

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
            {"table": WORKING_TABLE, "column": "original_term", "is_tgrm": True},
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
    # check: select * from biosym_annotations where original_term ~* '^[ ].*[ ]$';
    clean_up_junk()

    # big updates are much faster w/o this index, and it isn't needed from here on out anyway
    client.execute_query(
        "drop index trgm_index_biosym_annotations_original_term", ignore_error=True
    )

    remove_common_terms()  # remove one-off generic terms
    # remove_substrings()  # less specific terms in set with more specific terms # keeping substrings until we have ancestor search

    normalize_domains()

    # do this last to minimize mucking with attribute annotations
    client.select_insert_into_table(
        f"SELECT * from {SOURCE_TABLE} where domain='attributes'", WORKING_TABLE
    )


if __name__ == "__main__":
    """
    Checks:

    select sum(count) from (select count(*) as count from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and original_term<>'' group by lower(original_term) order by count(*) desc limit 1000) s;
    (556,711 -> 567,398 -> 908,930 -> 1,793,462)
    select sum(count) from (select count(*) as count from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and original_term<>'' group by lower(original_term) order by count(*) desc offset 10000) s;
    (2,555,158 -> 2,539,723 -> 3,697,848 -> 6,434,563)
    select count(*) from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and original_term<>'' and array_length(regexp_split_to_array(original_term, ' '), 1) > 1;
    (2,812,965 -> 2,786,428 -> 4,405,141 -> 7,265,719)
    select count(*) from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and original_term<>'';
    (3,748,417 -> 3,748,417 -> 5,552,648 -> 9,911,948)
    select domain, count(*) from biosym_annotations group by domain;
    attributes | 3032462
    compounds  | 1474950
    diseases   |  829121
    mechanisms | 1444346
    --
    attributes | 3721861
    compounds  | 3448590
    diseases   |  824851
    mechanisms | 5638507
    select sum(count) from (select original_term, count(*)  as count from biosym_annotations where original_term ilike '%inhibit%' group by original_term order by count(*) desc limit 100) s;
    (14,910 -> 15,206 -> 37,283 -> 34,083 -> 25,239)
    select sum(count) from (select original_term, count(*)  as count from biosym_annotations where original_term ilike '%inhibit%' group by original_term order by count(*) desc limit 1000) s;
    (38,315 -> 39,039 -> 76,872 -> 74,050 -> 59,714)
    select sum(count) from (select original_term, count(*)  as count from biosym_annotations where original_term ilike '%inhibit%' group by original_term order by count(*) desc offset 1000) s;
    (70,439 -> 69,715 -> 103,874 -> 165,806 -> 138,019)


    alter table terms ADD COLUMN id SERIAL PRIMARY KEY;
    DELETE FROM terms
    WHERE id IN
        (SELECT id
        FROM
            (SELECT id,
            ROW_NUMBER() OVER( PARTITION BY original_term, domain, character_offset_start, character_offset_end, publication_number
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
