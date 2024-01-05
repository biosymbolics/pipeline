"""
Patents ETL script
"""
import sys
import asyncio
import logging
from prisma.enums import BiomedicalEntityType
from prisma.models import Indicatable, Intervenable, Ownable, Patent


from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import (
    SOURCE_BIOSYM_ANNOTATIONS_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE,
)
from data.etl.biomedical_entity import BiomedicalEntityEtl
from data.etl.document import DocumentEtl
from data.etl.types import RelationConnectInfo, RelationIdFieldMap
from scripts.patents.constants import GPR_ANNOTATIONS_TABLE


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SOURCE_DB = "patents"

# save_json_as_file(terms, TERMS_FILE)
# domain == "attributes":
# if len(grouped_synonyms[0][1]) > MIN_CANONICAL_NAME_COUNT:
canonical_sql = "select id, canonical_name, preferred_name, instance_rollup, category_rollup from umls_lookup"
patents_annotation_sql = f"""
    SELECT term, domain, sum(count) as count FROM (
        -- TODO: can we prevent these from being matched to anything but diseases?
        SELECT lower(preferred_name) as term, domain, COUNT(*) as count
        from annotations
        group by preferred_name, domain

        UNION ALL

        SELECT lower(original_term) as term, domain, COUNT(*) as count
        FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
        group by original_term, domain
    ) s

    group by term, domain
"""

patents_annotations_2_sql = f"""
        SELECT publication_number,
            s.term as term,
            s.id as id,
            domain,
            max(source) as source,
            min(character_offset_start) as character_offset_start,
            min(character_offset_end) as character_offset_end,
            coalesce(max(t.instance_rollup), s.term) as instance_rollup,
            coalesce(max(t.category_rollup), s.term) as category_rollup
        from (
            SELECT
                publication_number,
                (CASE WHEN map.term is null THEN lower(original_term) ELSE map.term end) as term,
                (CASE WHEN map.id is null THEN lower(original_term) ELSE map.id end) as id,
                domain,
                source,
                character_offset_start,
                character_offset_end
                FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
                LEFT JOIN synonym_map map ON LOWER(original_term) = map.synonym
            ) s
            LEFT JOIN terms t ON s.id = t.id AND t.id <> ''
            group by publication_number, s.term, s.id, domain
        """

patents_from_gpr_sql = f"""
    SELECT publication_number,
        s.term as term,
        s.id as id,
        domain,
        max(source) as source,
        min(character_offset_start) as character_offset_start,
        min(character_offset_end) as character_offset_end,
        coalesce(max(t.instance_rollup), s.term) as instance_rollup,
        coalesce(max(t.category_rollup), s.term) as category_rollup
    from (
        SELECT
            publication_number,
            (CASE WHEN map.term is null THEN lower(preferred_name) ELSE map.term end) as term,
            (CASE WHEN map.id is null THEN lower(preferred_name) ELSE map.id end) as id,
            domain,
            source,
            character_offset_start,
            character_offset_end
            FROM {GPR_ANNOTATIONS_TABLE}
            LEFT JOIN synonym_map map ON LOWER(preferred_name) = map.synonym
        ) s
        LEFT JOIN terms t ON s.id = t.id AND t.id <> ''
        group by publication_number, s.term, s.id, domain
    """

patents_select_attributes_sql = f"""
    SELECT
        publication_number,
        original_term as term,
        original_term as id,
        domain,
        source,
        character_offset_start,
        character_offset_end,
        original_term as instance_rollup,
        original_term as category_rollup
    from {SOURCE_BIOSYM_ANNOTATIONS_TABLE}
    where domain='attributes'
"""


class PatentEtl(DocumentEtl):
    @staticmethod
    def get_source_sql(fields: list[str]):
        return f"""
            SELECT {", ".join(fields)}
            FROM
            applications
        """

    async def copy_indications(self):
        """
        Create indication records
        """
        fields = []
        source_records = await PsqlDatabaseClient(SOURCE_DB).select(
            query=PatentEtl.get_source_sql(fields)
        )

        source_map = {
            i: {"synonyms": [i], "default_type": BiomedicalEntityType.DISEASE}
            for sr in source_records
            for i in sr["indications"]
        }

        terms_to_insert = list(source_map.keys())
        terms_to_canonicalize = terms_to_insert

        await BiomedicalEntityEtl(
            "CompositeCandidateSelector",
            relation_id_field_map=RelationIdFieldMap(
                synonyms=RelationConnectInfo(
                    source_field="synonyms", dest_field="term", input_type="create"
                ),
            ),
        ).create_records(terms_to_canonicalize, terms_to_insert, source_map=source_map)

    async def copy_interventions(self):
        """
        Create intervention records
        """
        fields = []
        source_records = await PsqlDatabaseClient(SOURCE_DB).select(
            query=PatentEtl.get_source_sql(fields)
        )

        insert_map = {}

        terms_to_insert = list(insert_map.keys())
        terms_to_canonicalize = [
            k for k, v in insert_map.items() if v != BiomedicalEntityType.COMBINATION
        ]

        await BiomedicalEntityEtl(
            "CompositeCandidateSelector",
            relation_id_field_map=RelationIdFieldMap(
                synonyms=RelationConnectInfo(
                    source_field="synonyms", dest_field="term", input_type="create"
                ),
            ),
        ).create_records(
            terms_to_canonicalize,
            terms_to_insert,
            insert_map,
        )

    async def copy_documents(self):
        """
        Create regulatory approval records
        """
        fields = []
        patents = await PsqlDatabaseClient(SOURCE_DB).select(
            query=PatentEtl.get_source_sql(fields)
        )

        # create patent records
        await Patent.prisma().create_many(
            data=[],
            skip_duplicates=True,
        )

        # create assignee owner records
        await Ownable.prisma().create_many(
            data=[
                {
                    "name": a,
                    "is_primary": True,
                    "assignee_patent_id": p["id"],
                }
                for p in patents
                for a in p["assignees"]
            ],
            skip_duplicates=True,
        )

        # create inventor owner records
        await Ownable.prisma().create_many(
            data=[
                {
                    "name": i,
                    "is_primary": True,
                    "inventor_patent_id": p["id"],
                }
                for p in patents
                for i in p["invetors"]
            ],
            skip_duplicates=True,
        )

        # create "indicatable" records, those that map approval to a canonical indication
        await Indicatable.prisma().create_many(
            data=[{"name": "", "patent_id": p["id"]} for p in patents]
        )

        # create "intervenable" records, those that map approval to a canonical intervention
        await Intervenable.prisma().create_many(
            data=[
                {
                    "name": "",
                    "is_primary": True,
                    "regulatory_approval_id": p["id"],
                }
                for p in patents
            ]
        )


async def main():
    await PatentEtl(document_type="patent").copy_all()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.patents.copy_patents
            Copies patents data to biosym
            """
        )
        sys.exit()

    asyncio.run(main())
