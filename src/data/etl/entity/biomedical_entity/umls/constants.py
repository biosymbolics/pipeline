from prisma.enums import OntologyLevel


L1_CATEGORY_CUTOFF = 0.0001

ONTOLOGY_LEVEL_MAP = {
    OntologyLevel.SUBINSTANCE: 0,
    OntologyLevel.INSTANCE: 1,
    OntologyLevel.L1_CATEGORY: 2,
    OntologyLevel.L2_CATEGORY: 3,
    OntologyLevel.L3_CATEGORY: 4,
    OntologyLevel.L4_CATEGORY: 5,
    OntologyLevel.L5_CATEGORY: 6,
    OntologyLevel.NA: -1,
    OntologyLevel.UNKNOWN: -2,
}
