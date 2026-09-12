from scraper.infotecnica_api import (
    list_installations,
    find_installation_by_name,
    list_documents,
    find_diagram_document,
    download_document,
)

print("=== TEST STARTED ===")

try:
    print("\n[1] Getting substation list...")
    installations = list_installations("subestaciones")

    print(f"Found {len(installations)} installations")

    if installations:
        print("First installation:")
        print(installations[0])

except Exception as e:
    print("\nERROR in Step 1:")
    print(type(e).__name__, e)
    raise


try:
    print("\n[2] Searching for S/E CENTRAL ALFALFAL...")

    installation = find_installation_by_name(
        installations,
        "S/E CENTRAL ALFALFAL"
    )

    if installation is None:
        print("Installation NOT FOUND")
        raise SystemExit

    print("Installation FOUND:")
    print(installation)

except Exception as e:
    print("\nERROR in Step 2:")
    print(type(e).__name__, e)
    raise


try:
    print("\n[3] Getting documents...")

    documents = list_documents(
        "subestaciones",
        installation["id"]
    )

    print(f"Found {len(documents)} documents")

    for doc in documents[:10]:
        print(doc)

except Exception as e:
    print("\nERROR in Step 3:")
    print(type(e).__name__, e)
    raise


try:
    print("\n[4] Searching for diagram document...")

    diagram = find_diagram_document(documents)

    if diagram is None:
        print("NO DIAGRAM FOUND")
    else:
        print("DIAGRAM FOUND:")
        print(diagram)

except Exception as e:
    print("\nERROR in Step 4:")
    print(type(e).__name__, e)
    raise


if diagram is not None:

    try:
        print("\n[5] Downloading official document...")

        output_path = "test_downloads/" + diagram["filename"]

        print("Saving to:")
        print(output_path)

        downloaded = download_document(
            diagram,
            output_path
        )

        print("\nDOWNLOAD SUCCESSFUL")
        print(downloaded)

    except Exception as e:
        print("\nERROR in Step 5:")
        print(type(e).__name__, e)
        raise


print("\n=== TEST FINISHED ===")