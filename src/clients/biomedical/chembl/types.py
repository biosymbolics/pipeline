from typing import Optional, List, TypedDict


class MoleculeHierarchy(TypedDict):
    active_chembl_id: str
    molecule_chembl_id: str
    parent_chembl_id: str


class MoleculeProperties(TypedDict):
    alogp: str
    aromatic_rings: int
    cx_logd: str
    cx_logp: str
    cx_most_apka: str
    cx_most_bpka: str
    full_molformula: str
    full_mwt: str
    hba: int
    hba_lipinski: int
    hbd: int
    hbd_lipinski: int
    heavy_atoms: int
    molecular_species: str
    mw_freebase: str
    mw_monoisotopic: str
    np_likeness_score: str
    num_lipinski_ro5_violations: int
    num_ro5_violations: int
    psa: str
    qed_weighted: str
    ro3_pass: str
    rtb: int


class MoleculeStructures(TypedDict):
    canonical_smiles: str
    molfile: str
    standard_inchi: str
    standard_inchi_key: str


class MoleculeSynonym(TypedDict):
    molecule_synonym: str
    syn_type: str
    synonyms: str


class CrossReference(TypedDict):
    xref_id: str
    xref_name: Optional[str]
    xref_src: str


class ChemblMolecule(TypedDict):
    atc_classifications: List[str]
    availability_type: int
    biotherapeutic: Optional[str]
    black_box_warning: int
    chebi_par_id: Optional[str]
    chirality: int
    cross_references: List[CrossReference]
    dosed_ingredient: bool
    first_approval: Optional[str]
    first_in_class: int
    helm_notation: Optional[str]
    indication_class: Optional[str]
    inorganic_flag: int
    max_phase: str
    molecule_chembl_id: str
    molecule_hierarchy: MoleculeHierarchy
    molecule_properties: MoleculeProperties
    molecule_structures: MoleculeStructures
    molecule_synonyms: List[MoleculeSynonym]
    molecule_type: str
    natural_product: int
    oral: bool
    parenteral: bool
    polymer_flag: int
    pref_name: str
    prodrug: int
    score: float
    structure_type: str
    therapeutic_flag: bool
    topical: bool
    usan_stem: Optional[str]
    usan_stem_definition: Optional[str]
    usan_substem: Optional[str]
    usan_year: Optional[str]
    withdrawn_flag: bool


class ChemblResponse(TypedDict):
    molecules: List[ChemblMolecule]
