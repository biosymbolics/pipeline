"""
Patents ETL script
"""
import sys
import asyncio
import logging
from prisma.enums import BiomedicalEntityType
from prisma.models import Indicatable, Intervenable, Ownable, Patent


from clients.low_level.postgres import PsqlDatabaseClient
from data.etl.biomedical_entity import BiomedicalEntityEtl
from data.etl.document import DocumentEtl
from data.etl.types import RelationConnectInfo, RelationIdFieldMap


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SOURCE_DB = "patents"


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
