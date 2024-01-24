from prisma.enums import OntologyLevel

MAX_DENORMALIZED_ANCESTORS = 10

L1_CATEGORY_CUTOFF = 0.0001

ONTOLOGY_LEVEL_MAP = {
    OntologyLevel.INSTANCE: 1,
    OntologyLevel.L1_CATEGORY: 2,
    OntologyLevel.L2_CATEGORY: 3,
    # OntologyLevel.NA: -1, # excluded
    # OntologyLevel.UNKNOWN: -1, # excluded
}
