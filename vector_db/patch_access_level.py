from ingestor import get_collection


ACCESS_PUBLIC = "public"
ACCESS_JUNIOR = "junior"
ACCESS_SENIOR = "senior"
ACCESS_ADMIN  = "admin"

DEFAULT_LEVEL = ACCESS_JUNIOR 


RULES_BY_PROJECT: dict[str, list[tuple[str, list[str]]]] = {

    "scrapy": [
        (ACCESS_ADMIN, [
            "/scrapy/settings/",
            "/scrapy/telnet/",
        ]),
        (ACCESS_SENIOR, [
            "/scrapy/core/",
            "/scrapy/middleware/",
            "/scrapy/pipelines/",
            "/scrapy/http/",
            "/scrapy/extensions/",
        ]),
        (ACCESS_PUBLIC, [
            "/docs/",
            "/tests/",
            "/tests_typing/",
            "/extras/",
            "/sep/",
        ]),
        # Everything else - junior
    ],

    "freemed": [
        (ACCESS_ADMIN, [
            "/freemedsoftware/acl/",
            "/freemedsoftware/core/",
            "/lib/settings",        
            "/scripts/gnupg/",
            "/data/schema/",
        ]),
        (ACCESS_SENIOR, [
            "/freemedsoftware/api/",
            "/freemedsoftware/module/",
            "/services/",
            "/scripts/",
            "/data/source/",
        ]),
        (ACCESS_PUBLIC, [
            "/doc/",
            "/tests/",
            "/data/config/",
        ]),
    ],
}


def get_access_level(metadata: dict, project: str) -> str:
    rules = RULES_BY_PROJECT.get(project, [])
    path = metadata.get("file_path", "").lower()

    for level, patterns in rules:
        if any(pattern in path for pattern in patterns):
            return level

    return DEFAULT_LEVEL


BATCH_SIZE = 500  # stay well within ChromaDB's get() limits


def patch_project(project: str) -> None:

    if project not in RULES_BY_PROJECT:
        print(f"[patch] WARNING: No rules defined for project '{project}'. "
              f"All chunks will be tagged '{DEFAULT_LEVEL}'.")

    collection = get_collection()
    offset = 0
    total_patched = 0

    print(f"[patch] Starting access level patch for project: '{project}'")

    while True:
        results = collection.get(
            where={"project": {"$eq": project}},
            include=["metadatas"],
            limit=BATCH_SIZE,
            offset=offset,
        )
        if not results["ids"]:
            break

        updated_metadatas = [
            {**meta, "access_level": get_access_level(meta, project)}
            for meta in results["metadatas"]
        ]

        collection.update(ids=results["ids"], metadatas=updated_metadatas)

        total_patched += len(results["ids"])
        print(f"[patch]   {project}: {total_patched} chunks patched...")
        offset += len(results["ids"])

    print(f"[patch] Done. {total_patched} chunks updated for '{project}'.\n")


if __name__ == "__main__":
    patch_project("scrapy")
    patch_project("freemed")